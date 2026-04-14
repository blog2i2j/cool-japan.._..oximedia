//! Automated compliance recording scheduler with retention policy enforcement.
//!
//! Broadcast regulators (Ofcom, FCC, etc.) require that stations record their
//! outgoing programme feed continuously and retain those recordings for a
//! minimum period (often 28 or 60 days).  This module provides:
//!
//! * [`ComplianceRecorder`] — a stateful recorder that tracks the current
//!   programme segment being captured and manages the recording lifecycle.
//! * [`RecordingSchedule`] — a daily recording schedule specifying which hours
//!   require compliance capture.
//! * [`RetentionManager`] — enforces configurable retention windows, marking
//!   expired recordings for deletion without actually touching the filesystem
//!   (deletion is left to the operator so this module remains pure-logic).
//! * [`ComplianceReport`] — a summary report of the current compliance state.
//!
//! # Architecture
//!
//! ```text
//! PlayoutServer
//!   └─ ComplianceRecorder
//!        ├─ start_segment(id, pts_ns)  →  opens a RecordingSegment
//!        ├─ write_frame(pts_ns, bytes) →  accumulates byte count / CRC
//!        ├─ close_segment()            →  finalises and archives
//!        └─ enforce_retention(now)     →  delegates to RetentionManager
//! ```
//!
//! All operations are synchronous and `no_std`-friendly (aside from
//! `std::time::SystemTime` usage).

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors specific to the compliance recorder.
#[derive(Debug, Clone, PartialEq)]
pub enum ComplianceError {
    /// A segment is already open; close it before starting a new one.
    SegmentAlreadyOpen,
    /// No segment is currently open.
    NoOpenSegment,
    /// The segment was written without any frames — likely a configuration
    /// error.
    EmptySegment,
    /// Retention policy not found.
    PolicyNotFound(String),
    /// Invalid configuration value.
    InvalidConfig(String),
}

impl std::fmt::Display for ComplianceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SegmentAlreadyOpen => write!(f, "a recording segment is already open"),
            Self::NoOpenSegment => write!(f, "no recording segment is currently open"),
            Self::EmptySegment => write!(f, "segment closed with zero frames"),
            Self::PolicyNotFound(n) => write!(f, "retention policy '{n}' not found"),
            Self::InvalidConfig(m) => write!(f, "invalid configuration: {m}"),
        }
    }
}

impl std::error::Error for ComplianceError {}

/// Convenience result type.
pub type ComplianceResult<T> = Result<T, ComplianceError>;

// ---------------------------------------------------------------------------
// RetentionPolicy
// ---------------------------------------------------------------------------

/// How long a completed recording must be retained before it may be purged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetentionPolicy {
    /// Unique identifier / name for this policy.
    pub name: String,
    /// Minimum retention duration.
    pub min_duration: Duration,
    /// Whether to auto-archive (move to long-term cold storage) before
    /// purging.  The recorder records this intent but does not act on it.
    pub archive_before_purge: bool,
    /// Regulatory reference string (e.g. "Ofcom Rule 7(1)(a)").
    pub regulatory_ref: String,
}

impl RetentionPolicy {
    /// 28-day retention (typical Ofcom requirement).
    pub fn ofcom_28d() -> Self {
        Self {
            name: "ofcom-28d".to_string(),
            min_duration: Duration::from_hours(672), // 28 days
            archive_before_purge: false,
            regulatory_ref: "Ofcom Broadcasting Code Rule 7".to_string(),
        }
    }

    /// 60-day retention (FCC / extended compliance).
    pub fn fcc_60d() -> Self {
        Self {
            name: "fcc-60d".to_string(),
            min_duration: Duration::from_hours(1440), // 60 days
            archive_before_purge: true,
            regulatory_ref: "FCC 47 CFR §76.209(b)".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// RecordingSegment
// ---------------------------------------------------------------------------

/// Status of a compliance recording segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentStatus {
    /// Currently being written.
    Recording,
    /// Closed and waiting for retention window to expire.
    Retained,
    /// Retention window has expired; eligible for deletion.
    Expired,
    /// Segment has been purged from the catalogue.
    Purged,
}

/// A single compliance recording segment covering a contiguous programme span.
#[derive(Debug, Clone)]
pub struct RecordingSegment {
    /// Unique identifier for this segment (e.g. UUID or channel-timestamp).
    pub id: String,
    /// Programme title, if known.
    pub programme_title: String,
    /// Playout channel identifier.
    pub channel_id: String,
    /// Wall-clock time when the segment was opened.
    pub started_at: SystemTime,
    /// Wall-clock time when the segment was closed (`None` while recording).
    pub closed_at: Option<SystemTime>,
    /// First frame PTS in nanoseconds.
    pub start_pts_ns: u64,
    /// Last frame PTS in nanoseconds (`0` while recording).
    pub end_pts_ns: u64,
    /// Total number of frames written to this segment.
    pub frame_count: u64,
    /// Approximate byte count of all frame data written.
    pub byte_count: u64,
    /// Simple running CRC32 of all frame bytes (for quick integrity checks).
    pub checksum_crc32: u32,
    /// Current status of this segment.
    pub status: SegmentStatus,
    /// Name of the retention policy that governs this segment.
    pub retention_policy: String,
}

impl RecordingSegment {
    /// Duration of the recording (from start to close).  Returns `None` while
    /// still recording.
    pub fn duration(&self) -> Option<Duration> {
        self.closed_at.and_then(|c| c.duration_since(self.started_at).ok())
    }

    /// Return `true` if this segment may be purged given the supplied
    /// `RetentionPolicy` and the current time `now`.
    pub fn is_purgeable(&self, policy: &RetentionPolicy, now: SystemTime) -> bool {
        if self.status == SegmentStatus::Purged {
            return false; // already gone
        }
        let Some(closed_at) = self.closed_at else {
            return false; // still open
        };
        now.duration_since(closed_at)
            .map(|age| age >= policy.min_duration)
            .unwrap_or(false)
    }
}

// ---------------------------------------------------------------------------
// RecordingSchedule
// ---------------------------------------------------------------------------

/// A bitmask of hours (0–23) during which compliance recording is required.
///
/// Bit `i` in the inner `u32` corresponds to hour `i` of the day.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordingSchedule {
    hour_mask: u32,
}

impl RecordingSchedule {
    /// Always record (all 24 hours).
    pub fn always() -> Self {
        Self { hour_mask: 0x00FF_FFFF }
    }

    /// Never record (all hours disabled).
    pub fn never() -> Self {
        Self { hour_mask: 0 }
    }

    /// Create a schedule from an explicit set of active hours (0–23).
    pub fn from_hours(hours: &[u8]) -> Self {
        let mut mask = 0u32;
        for &h in hours {
            if h < 24 {
                mask |= 1 << h;
            }
        }
        Self { hour_mask: mask }
    }

    /// Return `true` if recording is required at the given hour (0–23).
    pub fn requires_recording(&self, hour: u8) -> bool {
        if hour >= 24 {
            return false;
        }
        (self.hour_mask >> hour) & 1 == 1
    }

    /// Number of hours that require recording.
    pub fn active_hours(&self) -> u32 {
        self.hour_mask.count_ones()
    }
}

// ---------------------------------------------------------------------------
// RetentionManager
// ---------------------------------------------------------------------------

/// Enforces retention policies across a collection of closed segments.
pub struct RetentionManager {
    policies: HashMap<String, RetentionPolicy>,
}

impl RetentionManager {
    pub fn new() -> Self {
        Self {
            policies: HashMap::new(),
        }
    }

    /// Register a retention policy.
    pub fn add_policy(&mut self, policy: RetentionPolicy) {
        self.policies.insert(policy.name.clone(), policy);
    }

    /// Look up a policy by name.
    pub fn policy(&self, name: &str) -> Option<&RetentionPolicy> {
        self.policies.get(name)
    }

    /// Evaluate all segments against their retention policies.
    ///
    /// Returns a list of `(segment_id, new_status)` pairs for segments whose
    /// status has changed.  The caller is responsible for applying the status
    /// change and, if desired, actually deleting the files.
    pub fn enforce(
        &self,
        segments: &[RecordingSegment],
        now: SystemTime,
    ) -> Vec<(String, SegmentStatus)> {
        let mut transitions = Vec::new();
        for seg in segments {
            if seg.status != SegmentStatus::Retained {
                continue;
            }
            let Some(policy) = self.policies.get(&seg.retention_policy) else {
                continue;
            };
            if seg.is_purgeable(policy, now) {
                transitions.push((seg.id.clone(), SegmentStatus::Expired));
            }
        }
        transitions
    }

    /// Apply a set of transitions returned by [`enforce`] to the segment list.
    pub fn apply_transitions(
        segments: &mut Vec<RecordingSegment>,
        transitions: &[(String, SegmentStatus)],
    ) {
        for (id, new_status) in transitions {
            for seg in segments.iter_mut() {
                if &seg.id == id {
                    seg.status = *new_status;
                    break;
                }
            }
        }
    }
}

impl Default for RetentionManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ComplianceRecorder
// ---------------------------------------------------------------------------

/// Configuration for the compliance recorder.
#[derive(Debug, Clone)]
pub struct RecorderConfig {
    /// Channel identifier (e.g. "BBC ONE", "CNN").
    pub channel_id: String,
    /// Name of the retention policy to attach to new segments.
    pub default_retention_policy: String,
    /// Recording schedule (which hours require capture).
    pub schedule: RecordingSchedule,
    /// Maximum segment duration before automatic rollover (in seconds).
    /// `0` means no automatic rollover.
    pub max_segment_secs: u64,
}

impl Default for RecorderConfig {
    fn default() -> Self {
        Self {
            channel_id: "DEFAULT".to_string(),
            default_retention_policy: "ofcom-28d".to_string(),
            schedule: RecordingSchedule::always(),
            max_segment_secs: 3600, // 1-hour segments
        }
    }
}

/// Automated compliance recorder.
///
/// Manages the open/close lifecycle of [`RecordingSegment`] objects and
/// delegates retention to an embedded [`RetentionManager`].
pub struct ComplianceRecorder {
    config: RecorderConfig,
    /// Currently-open segment, if any.
    current: Option<RecordingSegment>,
    /// Completed segments (retained or expired).
    archive: Vec<RecordingSegment>,
    /// Retention manager.
    retention: RetentionManager,
}

impl ComplianceRecorder {
    /// Create a new recorder with the given configuration.
    pub fn new(config: RecorderConfig) -> Self {
        let mut retention = RetentionManager::new();
        // Pre-register common policies.
        retention.add_policy(RetentionPolicy::ofcom_28d());
        retention.add_policy(RetentionPolicy::fcc_60d());
        Self {
            config,
            current: None,
            archive: Vec::new(),
            retention,
        }
    }

    /// Register a custom retention policy.
    pub fn add_policy(&mut self, policy: RetentionPolicy) {
        self.retention.add_policy(policy);
    }

    /// Start a new recording segment.
    ///
    /// Returns `Err(SegmentAlreadyOpen)` if a segment is currently open.
    pub fn start_segment(
        &mut self,
        id: impl Into<String>,
        programme_title: impl Into<String>,
        start_pts_ns: u64,
        started_at: SystemTime,
    ) -> ComplianceResult<()> {
        if self.current.is_some() {
            return Err(ComplianceError::SegmentAlreadyOpen);
        }
        self.current = Some(RecordingSegment {
            id: id.into(),
            programme_title: programme_title.into(),
            channel_id: self.config.channel_id.clone(),
            started_at,
            closed_at: None,
            start_pts_ns,
            end_pts_ns: 0,
            frame_count: 0,
            byte_count: 0,
            checksum_crc32: 0,
            status: SegmentStatus::Recording,
            retention_policy: self.config.default_retention_policy.clone(),
        });
        Ok(())
    }

    /// Record a frame's byte payload.
    ///
    /// `pts_ns` is the presentation timestamp of this frame.
    /// `data` is the encoded frame bytes (for checksum and byte-count
    /// accounting; not actually stored in memory beyond the CRC update).
    pub fn write_frame(&mut self, pts_ns: u64, data: &[u8]) -> ComplianceResult<()> {
        let seg = self.current.as_mut().ok_or(ComplianceError::NoOpenSegment)?;
        seg.frame_count += 1;
        seg.byte_count += data.len() as u64;
        seg.end_pts_ns = pts_ns;
        seg.checksum_crc32 = crc32_update(seg.checksum_crc32, data);
        Ok(())
    }

    /// Close the current segment.
    ///
    /// Returns `Err(EmptySegment)` if no frames were written.
    pub fn close_segment(&mut self, closed_at: SystemTime) -> ComplianceResult<RecordingSegment> {
        let mut seg = self.current.take().ok_or(ComplianceError::NoOpenSegment)?;
        if seg.frame_count == 0 {
            // Restore so caller can retry.
            self.current = Some(seg);
            return Err(ComplianceError::EmptySegment);
        }
        seg.closed_at = Some(closed_at);
        seg.status = SegmentStatus::Retained;
        let finished = seg.clone();
        self.archive.push(seg);
        Ok(finished)
    }

    /// Force-close the current segment even if empty (used on shutdown).
    ///
    /// Returns the closed segment (possibly empty), or `None` if no segment
    /// was open.
    pub fn force_close(&mut self, closed_at: SystemTime) -> Option<RecordingSegment> {
        let mut seg = self.current.take()?;
        seg.closed_at = Some(closed_at);
        seg.status = SegmentStatus::Retained;
        let finished = seg.clone();
        self.archive.push(seg);
        Some(finished)
    }

    /// Run retention enforcement at the given `now` timestamp.
    ///
    /// Returns the number of segments newly marked as expired.
    pub fn enforce_retention(&mut self, now: SystemTime) -> usize {
        let transitions = self.retention.enforce(&self.archive, now);
        let count = transitions.len();
        RetentionManager::apply_transitions(&mut self.archive, &transitions);
        count
    }

    /// Mark all expired segments as purged, returning their IDs.
    pub fn purge_expired(&mut self) -> Vec<String> {
        let mut purged = Vec::new();
        for seg in &mut self.archive {
            if seg.status == SegmentStatus::Expired {
                seg.status = SegmentStatus::Purged;
                purged.push(seg.id.clone());
            }
        }
        purged
    }

    /// Check whether recording is required right now given the schedule and a
    /// wall-clock hour (0–23).
    pub fn should_record_at_hour(&self, hour: u8) -> bool {
        self.config.schedule.requires_recording(hour)
    }

    /// Number of archived segments.
    pub fn archive_len(&self) -> usize {
        self.archive.len()
    }

    /// Return a reference to all archived segments.
    pub fn archive(&self) -> &[RecordingSegment] {
        &self.archive
    }

    /// Return `true` if a segment is currently open.
    pub fn is_recording(&self) -> bool {
        self.current.is_some()
    }

    /// Generate a compliance summary report.
    pub fn report(&self, now: SystemTime) -> ComplianceReport {
        let total = self.archive.len();
        let retained = self
            .archive
            .iter()
            .filter(|s| s.status == SegmentStatus::Retained)
            .count();
        let expired = self
            .archive
            .iter()
            .filter(|s| s.status == SegmentStatus::Expired)
            .count();
        let purged = self
            .archive
            .iter()
            .filter(|s| s.status == SegmentStatus::Purged)
            .count();
        let total_bytes: u64 = self.archive.iter().map(|s| s.byte_count).sum();
        let total_frames: u64 = self.archive.iter().map(|s| s.frame_count).sum();
        ComplianceReport {
            channel_id: self.config.channel_id.clone(),
            generated_at: now,
            total_segments: total,
            retained_segments: retained,
            expired_segments: expired,
            purged_segments: purged,
            total_bytes_recorded: total_bytes,
            total_frames_recorded: total_frames,
            currently_recording: self.current.is_some(),
        }
    }
}

// ---------------------------------------------------------------------------
// ComplianceReport
// ---------------------------------------------------------------------------

/// A point-in-time summary of the compliance recorder state.
#[derive(Debug, Clone)]
pub struct ComplianceReport {
    pub channel_id: String,
    pub generated_at: SystemTime,
    pub total_segments: usize,
    pub retained_segments: usize,
    pub expired_segments: usize,
    pub purged_segments: usize,
    pub total_bytes_recorded: u64,
    pub total_frames_recorded: u64,
    pub currently_recording: bool,
}

impl ComplianceReport {
    /// Return `true` if there are no expired (unpurged) segments.
    pub fn is_clean(&self) -> bool {
        self.expired_segments == 0
    }
}

// ---------------------------------------------------------------------------
// CRC-32 (IEEE 802.3) — pure-Rust running implementation
// ---------------------------------------------------------------------------

/// Update a running CRC-32 value with additional `data` bytes.
///
/// Uses the standard IEEE 802.3 polynomial (0xEDB88320 reflected).
pub fn crc32_update(mut crc: u32, data: &[u8]) -> u32 {
    crc = !crc;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// Compute the CRC-32 of a complete byte slice.
pub fn crc32(data: &[u8]) -> u32 {
    crc32_update(0, data)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_recorder() -> ComplianceRecorder {
        ComplianceRecorder::new(RecorderConfig::default())
    }

    fn now_plus_secs(secs: u64) -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000 + secs)
    }

    #[test]
    fn test_crc32_known_value() {
        // CRC-32 of "123456789" = 0xCBF43926 (standard test vector).
        let result = crc32(b"123456789");
        assert_eq!(result, 0xCBF4_3926, "CRC-32 standard vector failed");
    }

    #[test]
    fn test_crc32_incremental_matches_full() {
        let data = b"Hello, compliance recorder!";
        let full = crc32(data);
        let incremental = crc32_update(crc32_update(0, &data[..10]), &data[10..]);
        assert_eq!(full, incremental);
    }

    #[test]
    fn test_start_and_close_segment() {
        let mut rec = make_recorder();
        assert!(!rec.is_recording());

        rec.start_segment("seg-001", "News at Ten", 0, now_plus_secs(0))
            .expect("start should succeed");
        assert!(rec.is_recording());

        rec.write_frame(40_000_000, b"frame-data-1")
            .expect("write frame");
        rec.write_frame(80_000_000, b"frame-data-2")
            .expect("write frame");

        let seg = rec
            .close_segment(now_plus_secs(3600))
            .expect("close should succeed");
        assert_eq!(seg.frame_count, 2);
        assert_eq!(seg.status, SegmentStatus::Retained);
        assert!(!rec.is_recording());
        assert_eq!(rec.archive_len(), 1);
    }

    #[test]
    fn test_double_start_returns_error() {
        let mut rec = make_recorder();
        rec.start_segment("s1", "Prog A", 0, now_plus_secs(0))
            .unwrap();
        rec.write_frame(0, b"x").unwrap();
        let err = rec.start_segment("s2", "Prog B", 0, now_plus_secs(1));
        assert!(matches!(err, Err(ComplianceError::SegmentAlreadyOpen)));
    }

    #[test]
    fn test_close_empty_segment_returns_error() {
        let mut rec = make_recorder();
        rec.start_segment("s1", "Prog", 0, now_plus_secs(0)).unwrap();
        let err = rec.close_segment(now_plus_secs(1));
        assert!(matches!(err, Err(ComplianceError::EmptySegment)));
        // Recorder should still be recording (segment restored).
        assert!(rec.is_recording());
    }

    #[test]
    fn test_write_frame_without_open_segment() {
        let mut rec = make_recorder();
        let err = rec.write_frame(0, b"data");
        assert!(matches!(err, Err(ComplianceError::NoOpenSegment)));
    }

    #[test]
    fn test_retention_enforcement_marks_expired() {
        let mut rec = make_recorder();

        // Start and close a segment.
        rec.start_segment("s1", "Old Show", 0, now_plus_secs(0))
            .unwrap();
        rec.write_frame(0, b"frame").unwrap();
        rec.close_segment(now_plus_secs(1)).unwrap();

        // Enforcement now (retention not yet met) — no changes.
        let expired = rec.enforce_retention(now_plus_secs(100));
        assert_eq!(expired, 0);

        // Enforcement 29 days later (> 28-day Ofcom window).
        let future = now_plus_secs(29 * 24 * 3600 + 10);
        let expired = rec.enforce_retention(future);
        assert_eq!(expired, 1);

        let seg = &rec.archive()[0];
        assert_eq!(seg.status, SegmentStatus::Expired);
    }

    #[test]
    fn test_purge_expired_segments() {
        let mut rec = make_recorder();
        rec.start_segment("s1", "Show", 0, now_plus_secs(0))
            .unwrap();
        rec.write_frame(0, b"f").unwrap();
        rec.close_segment(now_plus_secs(1)).unwrap();

        rec.enforce_retention(now_plus_secs(29 * 24 * 3600));
        let purged = rec.purge_expired();
        assert_eq!(purged, vec!["s1".to_string()]);
        assert_eq!(rec.archive()[0].status, SegmentStatus::Purged);
    }

    #[test]
    fn test_recording_schedule_from_hours() {
        let sched = RecordingSchedule::from_hours(&[6, 12, 18]);
        assert!(sched.requires_recording(6));
        assert!(sched.requires_recording(12));
        assert!(sched.requires_recording(18));
        assert!(!sched.requires_recording(0));
        assert!(!sched.requires_recording(23));
        assert_eq!(sched.active_hours(), 3);
    }

    #[test]
    fn test_recording_schedule_always() {
        let sched = RecordingSchedule::always();
        for h in 0u8..24 {
            assert!(sched.requires_recording(h), "hour {h} should be active");
        }
        assert_eq!(sched.active_hours(), 24);
    }

    #[test]
    fn test_compliance_report() {
        let mut rec = make_recorder();
        // Archive 2 segments.
        for i in 0u64..2 {
            rec.start_segment(format!("s{i}"), "Prog", i * 3600_000_000_000, now_plus_secs(i * 3600))
                .unwrap();
            rec.write_frame(i * 40_000_000, b"data").unwrap();
            rec.close_segment(now_plus_secs(i * 3600 + 3600)).unwrap();
        }

        let report = rec.report(now_plus_secs(7200));
        assert_eq!(report.total_segments, 2);
        assert_eq!(report.retained_segments, 2);
        assert_eq!(report.expired_segments, 0);
        assert!(!report.currently_recording);
        assert!(report.is_clean());
        assert!(report.total_frames_recorded > 0);
    }

    #[test]
    fn test_force_close_empty_segment() {
        let mut rec = make_recorder();
        rec.start_segment("s1", "Test", 0, now_plus_secs(0)).unwrap();
        // No frames written.
        let seg = rec.force_close(now_plus_secs(10)).expect("force_close returns Some");
        assert_eq!(seg.frame_count, 0);
        assert_eq!(seg.status, SegmentStatus::Retained);
    }

    #[test]
    fn test_retention_policy_is_purgeable() {
        let policy = RetentionPolicy::ofcom_28d();
        let closed = now_plus_secs(0);
        let seg = RecordingSegment {
            id: "x".into(),
            programme_title: "X".into(),
            channel_id: "CH1".into(),
            started_at: closed,
            closed_at: Some(closed),
            start_pts_ns: 0,
            end_pts_ns: 0,
            frame_count: 1,
            byte_count: 10,
            checksum_crc32: 0,
            status: SegmentStatus::Retained,
            retention_policy: "ofcom-28d".into(),
        };

        // 1 day after close — not purgeable.
        assert!(!seg.is_purgeable(&policy, now_plus_secs(24 * 3600)));
        // 29 days after close — purgeable.
        assert!(seg.is_purgeable(&policy, now_plus_secs(29 * 24 * 3600)));
    }
}
