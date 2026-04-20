//! Compliance recording and ingest: record-on-air, signal verification,
//! and file delivery workflows.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

// ── Signal verification ───────────────────────────────────────────────────────

/// Status of a live signal
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalStatus {
    /// Signal present and within parameters
    Valid,
    /// Signal present but out of spec (levels, format mismatch, etc.)
    OutOfSpec,
    /// No signal detected
    Lost,
    /// Signal quality unknown / not yet assessed
    Unknown,
}

impl SignalStatus {
    pub fn is_ok(self) -> bool {
        self == Self::Valid
    }
}

/// Parameters used to verify a live video signal
#[derive(Debug, Clone)]
pub struct SignalVerificationConfig {
    /// Minimum acceptable video level (IRE / arbitrary units 0–100)
    pub min_video_level: f32,
    /// Maximum acceptable video level
    pub max_video_level: f32,
    /// Minimum audio loudness in LUFS
    pub min_loudness_lufs: f32,
    /// Maximum audio loudness in LUFS
    pub max_loudness_lufs: f32,
    /// Timeout before declaring signal lost (ms)
    pub signal_timeout_ms: u64,
}

impl Default for SignalVerificationConfig {
    fn default() -> Self {
        Self {
            min_video_level: 5.0,
            max_video_level: 100.0,
            min_loudness_lufs: -35.0,
            max_loudness_lufs: -10.0,
            signal_timeout_ms: 5000,
        }
    }
}

/// A snapshot of signal measurements
#[derive(Debug, Clone)]
pub struct SignalMeasurement {
    pub timestamp_ms: u64,
    pub video_level: f32,
    pub audio_loudness_lufs: f32,
    pub has_video: bool,
    pub has_audio: bool,
}

impl SignalMeasurement {
    pub fn new(
        timestamp_ms: u64,
        video_level: f32,
        audio_loudness_lufs: f32,
        has_video: bool,
        has_audio: bool,
    ) -> Self {
        Self {
            timestamp_ms,
            video_level,
            audio_loudness_lufs,
            has_video,
            has_audio,
        }
    }

    /// Verify measurement against config; returns the derived status
    pub fn verify(&self, cfg: &SignalVerificationConfig) -> SignalStatus {
        if !self.has_video {
            return SignalStatus::Lost;
        }
        if self.video_level < cfg.min_video_level || self.video_level > cfg.max_video_level {
            return SignalStatus::OutOfSpec;
        }
        if self.has_audio
            && (self.audio_loudness_lufs < cfg.min_loudness_lufs
                || self.audio_loudness_lufs > cfg.max_loudness_lufs)
        {
            return SignalStatus::OutOfSpec;
        }
        SignalStatus::Valid
    }
}

// ── Compliance recording ──────────────────────────────────────────────────────

/// Format of the compliance recording output
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplianceFormat {
    /// MPEG-2 Transport Stream
    Mpeg2Ts,
    /// MXF OP1a
    MxfOp1a,
    /// MP4 / ISOBMFF
    Mp4,
    /// Raw PCM + H.264 in MKV container
    MkvH264,
}

impl ComplianceFormat {
    pub fn extension(self) -> &'static str {
        match self {
            Self::Mpeg2Ts => "ts",
            Self::MxfOp1a => "mxf",
            Self::Mp4 => "mp4",
            Self::MkvH264 => "mkv",
        }
    }
}

/// State of a compliance recording session
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplianceRecordingState {
    Idle,
    Arming,
    Recording,
    Stopping,
    Completed,
    Error,
}

impl ComplianceRecordingState {
    pub fn is_active(self) -> bool {
        matches!(self, Self::Arming | Self::Recording)
    }
}

/// A compliance recording session (record-on-air log)
#[derive(Debug, Clone)]
pub struct ComplianceRecording {
    pub id: String,
    pub channel_id: String,
    pub format: ComplianceFormat,
    pub state: ComplianceRecordingState,
    pub start_time: Option<SystemTime>,
    pub stop_time: Option<SystemTime>,
    pub output_path: String,
    /// Bytes written so far
    pub bytes_written: u64,
    /// Number of signal anomalies detected during recording
    pub anomaly_count: u32,
}

impl ComplianceRecording {
    pub fn new(id: &str, channel_id: &str, format: ComplianceFormat, output_dir: &str) -> Self {
        let output_path = format!("{}/{}.{}", output_dir, id, format.extension());
        Self {
            id: id.to_string(),
            channel_id: channel_id.to_string(),
            format,
            state: ComplianceRecordingState::Idle,
            start_time: None,
            stop_time: None,
            output_path,
            bytes_written: 0,
            anomaly_count: 0,
        }
    }

    /// Arm the recording (transition Idle → Arming)
    pub fn arm(&mut self) -> bool {
        if self.state == ComplianceRecordingState::Idle {
            self.state = ComplianceRecordingState::Arming;
            true
        } else {
            false
        }
    }

    /// Start the recording (transition Arming → Recording)
    pub fn start(&mut self, now: SystemTime) -> bool {
        if self.state == ComplianceRecordingState::Arming {
            self.state = ComplianceRecordingState::Recording;
            self.start_time = Some(now);
            true
        } else {
            false
        }
    }

    /// Append bytes to the recording
    pub fn write_bytes(&mut self, count: u64) {
        self.bytes_written += count;
    }

    /// Record a signal anomaly
    pub fn record_anomaly(&mut self) {
        self.anomaly_count += 1;
    }

    /// Stop the recording
    pub fn stop(&mut self, now: SystemTime) -> bool {
        if self.state == ComplianceRecordingState::Recording {
            self.state = ComplianceRecordingState::Stopping;
            self.stop_time = Some(now);
            self.state = ComplianceRecordingState::Completed;
            true
        } else {
            false
        }
    }

    /// Duration of the recording (if both start and stop are known)
    pub fn recorded_duration(&self) -> Option<Duration> {
        match (self.start_time, self.stop_time) {
            (Some(start), Some(stop)) => stop.duration_since(start).ok(),
            _ => None,
        }
    }
}

// ── File delivery ─────────────────────────────────────────────────────────────

/// Delivery destination type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeliveryDestination {
    /// Local filesystem path
    LocalPath(String),
    /// FTP/SFTP remote
    RemoteSftp { host: String, path: String },
    /// S3-compatible object storage
    S3 { bucket: String, key: String },
}

/// State of a file delivery job
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliveryState {
    Queued,
    Transferring,
    Verifying,
    Completed,
    Failed,
}

/// A file delivery job
#[derive(Debug, Clone)]
pub struct DeliveryJob {
    pub id: String,
    pub source_path: String,
    pub destination: DeliveryDestination,
    pub state: DeliveryState,
    pub bytes_transferred: u64,
    pub total_bytes: u64,
    pub retry_count: u32,
    pub max_retries: u32,
}

impl DeliveryJob {
    pub fn new(
        id: &str,
        source_path: &str,
        destination: DeliveryDestination,
        total_bytes: u64,
    ) -> Self {
        Self {
            id: id.to_string(),
            source_path: source_path.to_string(),
            destination,
            state: DeliveryState::Queued,
            bytes_transferred: 0,
            total_bytes,
            retry_count: 0,
            max_retries: 3,
        }
    }

    /// Progress as a fraction 0.0–1.0
    #[allow(clippy::cast_precision_loss)]
    pub fn progress(&self) -> f32 {
        if self.total_bytes == 0 {
            return 1.0;
        }
        self.bytes_transferred as f32 / self.total_bytes as f32
    }

    /// Simulate transferring a chunk
    pub fn transfer_chunk(&mut self, bytes: u64) {
        self.bytes_transferred = (self.bytes_transferred + bytes).min(self.total_bytes);
        if self.bytes_transferred >= self.total_bytes {
            self.state = DeliveryState::Verifying;
        } else if self.state == DeliveryState::Queued {
            self.state = DeliveryState::Transferring;
        }
    }

    /// Mark delivery as complete
    pub fn complete(&mut self) {
        self.state = DeliveryState::Completed;
    }

    /// Mark delivery as failed, incrementing retry counter
    pub fn fail(&mut self) {
        self.retry_count += 1;
        if self.retry_count >= self.max_retries {
            self.state = DeliveryState::Failed;
        } else {
            self.state = DeliveryState::Queued; // re-queue for retry
        }
    }

    pub fn can_retry(&self) -> bool {
        self.retry_count < self.max_retries
    }
}

// ── Retention policy ──────────────────────────────────────────────────────────

/// Rule that governs how long compliance recordings are kept on disk.
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Human-readable policy name.
    pub name: String,
    /// How long to retain recordings (0 = keep forever).
    pub retain_duration: Duration,
    /// Maximum storage consumed by this policy tier (bytes, 0 = unlimited).
    pub max_storage_bytes: u64,
    /// Whether to automatically delete recordings that exceed the retain window.
    pub auto_delete: bool,
    /// Destination for archival before deletion (empty = discard).
    pub archive_path: String,
}

impl RetentionPolicy {
    /// Create a "keep for N days" policy.
    pub fn keep_for_days(name: &str, days: u64, auto_delete: bool) -> Self {
        Self {
            name: name.to_string(),
            retain_duration: Duration::from_secs(days * 86_400),
            max_storage_bytes: 0,
            auto_delete,
            archive_path: String::new(),
        }
    }

    /// Create a "keep forever" policy.
    pub fn keep_forever(name: &str) -> Self {
        Self {
            name: name.to_string(),
            retain_duration: Duration::ZERO,
            max_storage_bytes: 0,
            auto_delete: false,
            archive_path: String::new(),
        }
    }

    /// Check whether a recording has exceeded the retention window.
    ///
    /// Returns `true` if the recording should be deleted / archived.
    /// A zero `retain_duration` means keep forever → always returns `false`.
    pub fn is_expired(&self, stop_time: SystemTime, now: SystemTime) -> bool {
        if self.retain_duration == Duration::ZERO {
            return false;
        }
        match now.duration_since(stop_time) {
            Ok(age) => age > self.retain_duration,
            Err(_) => false, // stop_time is in the future
        }
    }
}

/// Result of a retention enforcement run.
#[derive(Debug, Clone, Default)]
pub struct RetentionEnforcementResult {
    /// IDs of recordings that were marked for deletion.
    pub expired_ids: Vec<String>,
    /// IDs of recordings that were archived before deletion.
    pub archived_ids: Vec<String>,
    /// Total bytes that would be reclaimed.
    pub bytes_to_reclaim: u64,
}

// ── Compliance ingest coordinator ─────────────────────────────────────────────

/// Central coordinator for compliance recording and file delivery
#[derive(Debug, Default)]
pub struct ComplianceIngestCoordinator {
    recordings: HashMap<String, ComplianceRecording>,
    delivery_queue: Vec<DeliveryJob>,
    verification_config: SignalVerificationConfig,
    /// Named retention policies (policy name → policy).
    retention_policies: HashMap<String, RetentionPolicy>,
    /// Mapping from recording ID to its assigned policy name (empty = default).
    recording_policies: HashMap<String, String>,
}

impl ComplianceIngestCoordinator {
    pub fn new() -> Self {
        Self {
            verification_config: SignalVerificationConfig::default(),
            ..Default::default()
        }
    }

    pub fn with_verification_config(mut self, cfg: SignalVerificationConfig) -> Self {
        self.verification_config = cfg;
        self
    }

    /// Register a compliance recording
    pub fn register(&mut self, rec: ComplianceRecording) {
        self.recordings.insert(rec.id.clone(), rec);
    }

    /// Register a compliance recording and associate it with a retention policy.
    pub fn register_with_policy(&mut self, rec: ComplianceRecording, policy_name: &str) {
        self.recording_policies
            .insert(rec.id.clone(), policy_name.to_string());
        self.recordings.insert(rec.id.clone(), rec);
    }

    /// Get a mutable reference to a recording
    pub fn recording_mut(&mut self, id: &str) -> Option<&mut ComplianceRecording> {
        self.recordings.get_mut(id)
    }

    /// Enqueue a file delivery job
    pub fn enqueue_delivery(&mut self, job: DeliveryJob) {
        self.delivery_queue.push(job);
    }

    /// Verify a signal measurement against the configured thresholds
    pub fn verify_signal(&self, measurement: &SignalMeasurement) -> SignalStatus {
        measurement.verify(&self.verification_config)
    }

    /// Count recordings in a given state
    pub fn recording_count_by_state(&self, state: ComplianceRecordingState) -> usize {
        self.recordings
            .values()
            .filter(|r| r.state == state)
            .count()
    }

    /// Total bytes written across all recordings
    pub fn total_bytes_recorded(&self) -> u64 {
        self.recordings.values().map(|r| r.bytes_written).sum()
    }

    /// Return all delivery jobs that need attention (queued or transferring)
    pub fn active_deliveries(&self) -> Vec<&DeliveryJob> {
        self.delivery_queue
            .iter()
            .filter(|j| matches!(j.state, DeliveryState::Queued | DeliveryState::Transferring))
            .collect()
    }

    // ── Retention policy management ──────────────────────────────────────────

    /// Add a named retention policy.
    pub fn add_policy(&mut self, policy: RetentionPolicy) {
        self.retention_policies.insert(policy.name.clone(), policy);
    }

    /// Remove a retention policy by name.
    pub fn remove_policy(&mut self, name: &str) -> bool {
        self.retention_policies.remove(name).is_some()
    }

    /// Look up a retention policy by name.
    pub fn policy(&self, name: &str) -> Option<&RetentionPolicy> {
        self.retention_policies.get(name)
    }

    /// Evaluate all completed recordings against their retention policies.
    ///
    /// Returns a `RetentionEnforcementResult` describing which recordings have
    /// expired.  Does NOT actually delete any files; the caller is responsible
    /// for invoking file-system operations based on the returned IDs.
    pub fn enforce_retention(&self, now: SystemTime) -> RetentionEnforcementResult {
        let mut result = RetentionEnforcementResult::default();

        for rec in self.recordings.values() {
            // Only evaluate completed recordings.
            if rec.state != ComplianceRecordingState::Completed {
                continue;
            }
            let stop_time = match rec.stop_time {
                Some(t) => t,
                None => continue,
            };

            let policy_name = self
                .recording_policies
                .get(&rec.id)
                .map(String::as_str)
                .unwrap_or("");

            let expired = if policy_name.is_empty() {
                // No policy: never expires.
                false
            } else {
                match self.retention_policies.get(policy_name) {
                    Some(p) => p.is_expired(stop_time, now),
                    None => false,
                }
            };

            if expired {
                if let Some(policy) = self.retention_policies.get(policy_name) {
                    if !policy.archive_path.is_empty() {
                        result.archived_ids.push(rec.id.clone());
                    }
                }
                result.expired_ids.push(rec.id.clone());
                result.bytes_to_reclaim += rec.bytes_written;
            }
        }

        result
    }

    /// Purge recordings from the registry that have been marked as expired by
    /// `enforce_retention`.  Returns the number of recordings removed.
    pub fn purge_expired(&mut self, expired_ids: &[String]) -> usize {
        let before = self.recordings.len();
        for id in expired_ids {
            self.recordings.remove(id.as_str());
            self.recording_policies.remove(id.as_str());
        }
        before - self.recordings.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_str(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("oximedia-playout-compliance-{name}"))
            .to_string_lossy()
            .into_owned()
    }

    fn tmp_dir() -> String {
        std::env::temp_dir().to_string_lossy().into_owned()
    }

    #[test]
    fn test_signal_status_is_ok() {
        assert!(SignalStatus::Valid.is_ok());
        assert!(!SignalStatus::Lost.is_ok());
    }

    #[test]
    fn test_signal_measurement_verify_valid() {
        let m = SignalMeasurement::new(0, 75.0, -20.0, true, true);
        let status = m.verify(&SignalVerificationConfig::default());
        assert_eq!(status, SignalStatus::Valid);
    }

    #[test]
    fn test_signal_measurement_verify_out_of_spec_video() {
        let m = SignalMeasurement::new(0, 1.0, -20.0, true, true); // below min
        let status = m.verify(&SignalVerificationConfig::default());
        assert_eq!(status, SignalStatus::OutOfSpec);
    }

    #[test]
    fn test_signal_measurement_verify_lost() {
        let m = SignalMeasurement::new(0, 0.0, 0.0, false, false);
        let status = m.verify(&SignalVerificationConfig::default());
        assert_eq!(status, SignalStatus::Lost);
    }

    #[test]
    fn test_compliance_format_extension() {
        assert_eq!(ComplianceFormat::Mpeg2Ts.extension(), "ts");
        assert_eq!(ComplianceFormat::MxfOp1a.extension(), "mxf");
    }

    #[test]
    fn test_compliance_recording_arm_start_stop() {
        let dir = tmp_dir();
        let mut rec = ComplianceRecording::new("r1", "ch1", ComplianceFormat::Mpeg2Ts, &dir);
        assert!(rec.arm());
        assert!(rec.start(SystemTime::UNIX_EPOCH));
        rec.write_bytes(1024 * 1024);
        assert!(rec.stop(SystemTime::UNIX_EPOCH + Duration::from_mins(1)));
        assert_eq!(rec.state, ComplianceRecordingState::Completed);
        assert_eq!(rec.bytes_written, 1024 * 1024);
    }

    #[test]
    fn test_compliance_recording_duration() {
        let dir = tmp_dir();
        let mut rec = ComplianceRecording::new("r1", "ch1", ComplianceFormat::Mp4, &dir);
        rec.arm();
        rec.start(SystemTime::UNIX_EPOCH);
        rec.stop(SystemTime::UNIX_EPOCH + Duration::from_hours(1));
        let dur = rec.recorded_duration().expect("should succeed in test");
        assert_eq!(dur.as_secs(), 3600);
    }

    #[test]
    fn test_compliance_recording_anomaly_count() {
        let dir = tmp_dir();
        let mut rec = ComplianceRecording::new("r1", "ch1", ComplianceFormat::MxfOp1a, &dir);
        rec.record_anomaly();
        rec.record_anomaly();
        assert_eq!(rec.anomaly_count, 2);
    }

    #[test]
    fn test_delivery_job_progress() {
        let src = tmp_str("r1.ts");
        let mut job = DeliveryJob::new(
            "d1",
            &src,
            DeliveryDestination::LocalPath("/archive/r1.ts".to_string()),
            1000,
        );
        job.transfer_chunk(500);
        assert!((job.progress() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_delivery_job_complete() {
        let src = tmp_str("r1.ts");
        let mut job = DeliveryJob::new(
            "d1",
            &src,
            DeliveryDestination::LocalPath("/archive/r1.ts".to_string()),
            1000,
        );
        job.transfer_chunk(1000);
        job.complete();
        assert_eq!(job.state, DeliveryState::Completed);
    }

    #[test]
    fn test_delivery_job_retry_logic() {
        let src = tmp_str("r1.ts");
        let mut job = DeliveryJob::new(
            "d1",
            &src,
            DeliveryDestination::LocalPath("/x".to_string()),
            100,
        );
        job.fail(); // retry 1
        assert_eq!(job.state, DeliveryState::Queued);
        job.fail(); // retry 2
        assert_eq!(job.state, DeliveryState::Queued);
        job.fail(); // retry 3 — max reached
        assert_eq!(job.state, DeliveryState::Failed);
    }

    #[test]
    fn test_coordinator_register_and_count() {
        let mut coord = ComplianceIngestCoordinator::new();
        let dir = tmp_dir();
        let mut rec = ComplianceRecording::new("r1", "ch1", ComplianceFormat::Mp4, &dir);
        rec.arm();
        rec.start(SystemTime::UNIX_EPOCH);
        coord.register(rec);
        assert_eq!(
            coord.recording_count_by_state(ComplianceRecordingState::Recording),
            1
        );
    }

    #[test]
    fn test_coordinator_total_bytes() {
        let mut coord = ComplianceIngestCoordinator::new();
        let dir = tmp_dir();
        let mut rec = ComplianceRecording::new("r1", "ch1", ComplianceFormat::Mpeg2Ts, &dir);
        rec.write_bytes(500_000);
        coord.register(rec);
        assert_eq!(coord.total_bytes_recorded(), 500_000);
    }

    #[test]
    fn test_coordinator_active_deliveries() {
        let mut coord = ComplianceIngestCoordinator::new();
        let src = tmp_str("f.ts");
        let job = DeliveryJob::new(
            "d1",
            &src,
            DeliveryDestination::LocalPath("/arch".to_string()),
            100,
        );
        coord.enqueue_delivery(job);
        assert_eq!(coord.active_deliveries().len(), 1);
    }
}
