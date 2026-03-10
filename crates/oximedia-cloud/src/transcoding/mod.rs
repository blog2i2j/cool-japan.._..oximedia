//! Cloud transcoding job management
//!
//! Provides types and utilities for submitting, tracking, and costing cloud-based
//! media transcoding jobs without making any real network calls.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// Status of a cloud transcoding job
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CloudJobStatus {
    /// Waiting in the queue
    Queued,
    /// Currently being processed
    Processing,
    /// Finished successfully
    Completed,
    /// Ended with an error
    Failed,
    /// Cancelled by the user
    Cancelled,
}

impl CloudJobStatus {
    /// Returns `true` if the job will not change state any further.
    #[must_use]
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            CloudJobStatus::Completed | CloudJobStatus::Failed | CloudJobStatus::Cancelled
        )
    }

    /// Human-readable label
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            CloudJobStatus::Queued => "Queued",
            CloudJobStatus::Processing => "Processing",
            CloudJobStatus::Completed => "Completed",
            CloudJobStatus::Failed => "Failed",
            CloudJobStatus::Cancelled => "Cancelled",
        }
    }
}

/// A single cloud transcoding job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudTranscodeJob {
    /// Unique job identifier
    pub id: String,
    /// URL or path of the source media
    pub input_url: String,
    /// Cloud storage prefix where outputs are written
    pub output_prefix: String,
    /// Names of the presets to apply
    pub presets: Vec<String>,
    /// Current job status
    pub status: CloudJobStatus,
    /// Error message, set when status is `Failed`
    pub error_message: Option<String>,
    /// Estimated duration of the source in seconds
    pub source_duration_secs: f64,
}

impl CloudTranscodeJob {
    /// Create a new queued job
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        input_url: impl Into<String>,
        output_prefix: impl Into<String>,
        presets: Vec<String>,
        source_duration_secs: f64,
    ) -> Self {
        Self {
            id: id.into(),
            input_url: input_url.into(),
            output_prefix: output_prefix.into(),
            presets,
            status: CloudJobStatus::Queued,
            error_message: None,
            source_duration_secs,
        }
    }

    /// Transition to Processing
    pub fn start(&mut self) {
        self.status = CloudJobStatus::Processing;
    }

    /// Transition to Completed
    pub fn complete(&mut self) {
        self.status = CloudJobStatus::Completed;
    }

    /// Transition to Failed with a message
    pub fn fail(&mut self, reason: impl Into<String>) {
        self.status = CloudJobStatus::Failed;
        self.error_message = Some(reason.into());
    }

    /// Transition to Cancelled
    pub fn cancel(&mut self) {
        self.status = CloudJobStatus::Cancelled;
    }
}

/// A reusable transcoding preset (codec, bitrate, resolution)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudTranscodePreset {
    /// Unique preset name
    pub name: String,
    /// Video codec identifier (e.g. "h264", "hevc", "vp9")
    pub video_codec: String,
    /// Audio codec identifier (e.g. "aac", "opus")
    pub audio_codec: String,
    /// Target video bitrate in kbps (0 for audio-only)
    pub bitrate_kbps: u32,
    /// Output resolution as (width, height) in pixels
    pub resolution: (u32, u32),
}

impl CloudTranscodePreset {
    /// Create a custom preset
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        video_codec: impl Into<String>,
        audio_codec: impl Into<String>,
        bitrate_kbps: u32,
        resolution: (u32, u32),
    ) -> Self {
        Self {
            name: name.into(),
            video_codec: video_codec.into(),
            audio_codec: audio_codec.into(),
            bitrate_kbps,
            resolution,
        }
    }

    /// 1080p HD preset (H.264 / AAC, 5 Mbps)
    #[must_use]
    pub fn hd_1080p() -> Self {
        Self::new("hd_1080p", "h264", "aac", 5_000, (1920, 1080))
    }

    /// 720p HD preset (H.264 / AAC, 2.5 Mbps)
    #[must_use]
    pub fn hd_720p() -> Self {
        Self::new("hd_720p", "h264", "aac", 2_500, (1280, 720))
    }

    /// 480p SD preset (H.264 / AAC, 1 Mbps)
    #[must_use]
    pub fn sd_480p() -> Self {
        Self::new("sd_480p", "h264", "aac", 1_000, (854, 480))
    }

    /// 360p mobile preset (H.264 / AAC, 500 kbps)
    #[must_use]
    pub fn mobile_360p() -> Self {
        Self::new("mobile_360p", "h264", "aac", 500, (640, 360))
    }

    /// Audio-only preset (AAC, 128 kbps, no video)
    #[must_use]
    pub fn audio_only() -> Self {
        Self::new("audio_only", "", "aac", 128, (0, 0))
    }

    /// Whether this preset produces a video stream
    #[must_use]
    pub fn has_video(&self) -> bool {
        !self.video_codec.is_empty() && self.resolution != (0, 0)
    }

    /// Pixel count (width × height)
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        self.resolution.0 as u64 * self.resolution.1 as u64
    }
}

/// An entry in the priority queue
#[derive(Debug, Clone)]
struct QueueEntry {
    job: CloudTranscodeJob,
    /// Higher value = higher priority
    priority: u8,
}

/// Priority queue for cloud transcoding jobs.
///
/// Jobs are stored sorted in descending priority order; the highest-priority
/// job is always at index 0.
pub struct CloudTranscodeQueue {
    entries: Vec<QueueEntry>,
}

impl CloudTranscodeQueue {
    /// Create an empty queue
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Submit a job with a given priority (0 = lowest, 255 = highest).
    pub fn submit(&mut self, job: CloudTranscodeJob, priority: u8) {
        let entry = QueueEntry { job, priority };
        // Insert at the correct position to keep the vec sorted descending
        let pos = self.entries.partition_point(|e| e.priority >= priority);
        self.entries.insert(pos, entry);
    }

    /// Remove and return the next (highest-priority) job, or `None` if empty.
    pub fn next(&mut self) -> Option<CloudTranscodeJob> {
        if self.entries.is_empty() {
            None
        } else {
            Some(self.entries.remove(0).job)
        }
    }

    /// Number of jobs currently waiting
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the queue is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Peek at the next job without removing it
    #[must_use]
    pub fn peek(&self) -> Option<&CloudTranscodeJob> {
        self.entries.first().map(|e| &e.job)
    }
}

impl Default for CloudTranscodeQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Accumulated metrics for a transcoding queue or service
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JobMetrics {
    /// Total jobs ever submitted
    pub total_submitted: u64,
    /// Jobs that completed successfully
    pub total_completed: u64,
    /// Jobs that failed
    pub total_failed: u64,
    /// Rolling average job duration in seconds
    pub avg_duration_secs: f32,
}

impl JobMetrics {
    /// Create new zeroed metrics
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a job submission
    pub fn record_submitted(&mut self) {
        self.total_submitted += 1;
    }

    /// Record a successful completion with its duration
    pub fn record_completed(&mut self, duration_secs: f32) {
        let prev_total = self.total_completed as f32;
        self.avg_duration_secs =
            (self.avg_duration_secs * prev_total + duration_secs) / (prev_total + 1.0);
        self.total_completed += 1;
    }

    /// Record a failure
    pub fn record_failed(&mut self) {
        self.total_failed += 1;
    }

    /// Success rate (0.0 – 1.0).  Returns 0.0 if nothing has been submitted.
    #[must_use]
    pub fn success_rate(&self) -> f32 {
        let terminal = self.total_completed + self.total_failed;
        if terminal == 0 {
            0.0
        } else {
            self.total_completed as f32 / terminal as f32
        }
    }
}

/// Estimates the cost of cloud transcoding jobs based on simplified provider
/// pricing models.
pub struct CostEstimator {
    /// Per-minute cost for HD output (≥ 720p)
    hd_price_per_minute: f64,
    /// Per-minute cost for SD output (< 720p)
    sd_price_per_minute: f64,
    /// Per-minute cost for audio-only output
    audio_price_per_minute: f64,
}

impl CostEstimator {
    /// Create an estimator with AWS Elemental MediaConvert-like pricing (USD)
    #[must_use]
    pub fn new() -> Self {
        Self {
            hd_price_per_minute: 0.024,    // ~$0.024 / min for HD
            sd_price_per_minute: 0.017,    // ~$0.017 / min for SD
            audio_price_per_minute: 0.004, // ~$0.004 / min for audio
        }
    }

    /// Create an estimator with custom per-minute prices
    #[must_use]
    pub fn with_prices(
        hd_price_per_minute: f64,
        sd_price_per_minute: f64,
        audio_price_per_minute: f64,
    ) -> Self {
        Self {
            hd_price_per_minute,
            sd_price_per_minute,
            audio_price_per_minute,
        }
    }

    /// Estimate the cost (USD) for transcoding a piece of media.
    ///
    /// `resolution` is (width, height); pass `(0, 0)` for audio-only output.
    #[must_use]
    pub fn estimate_transcode_cost(&self, duration_secs: f64, resolution: (u32, u32)) -> f64 {
        let minutes = duration_secs / 60.0;
        let price_per_min = if resolution == (0, 0) {
            self.audio_price_per_minute
        } else if resolution.1 >= 720 {
            self.hd_price_per_minute
        } else {
            self.sd_price_per_minute
        };
        minutes * price_per_min
    }

    /// Estimate total cost for a job across all its presets
    #[must_use]
    pub fn estimate_job_cost(
        &self,
        job: &CloudTranscodeJob,
        presets: &[CloudTranscodePreset],
    ) -> f64 {
        presets
            .iter()
            .filter(|p| job.presets.contains(&p.name))
            .map(|p| self.estimate_transcode_cost(job.source_duration_secs, p.resolution))
            .sum()
    }
}

impl Default for CostEstimator {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_job(id: &str) -> CloudTranscodeJob {
        CloudTranscodeJob::new(
            id,
            "s3://bucket/input.mp4",
            "s3://bucket/output/",
            vec!["hd_1080p".to_string(), "hd_720p".to_string()],
            3600.0, // 1-hour source
        )
    }

    // 1. CloudJobStatus::is_terminal()
    #[test]
    fn test_job_status_terminal() {
        assert!(!CloudJobStatus::Queued.is_terminal());
        assert!(!CloudJobStatus::Processing.is_terminal());
        assert!(CloudJobStatus::Completed.is_terminal());
        assert!(CloudJobStatus::Failed.is_terminal());
        assert!(CloudJobStatus::Cancelled.is_terminal());
    }

    // 2. CloudJobStatus::label()
    #[test]
    fn test_job_status_label() {
        assert_eq!(CloudJobStatus::Queued.label(), "Queued");
        assert_eq!(CloudJobStatus::Processing.label(), "Processing");
        assert_eq!(CloudJobStatus::Completed.label(), "Completed");
    }

    // 3. CloudTranscodeJob lifecycle transitions
    #[test]
    fn test_job_lifecycle() {
        let mut job = make_job("job-1");
        assert_eq!(job.status, CloudJobStatus::Queued);
        job.start();
        assert_eq!(job.status, CloudJobStatus::Processing);
        job.complete();
        assert_eq!(job.status, CloudJobStatus::Completed);
    }

    // 4. Job failure message
    #[test]
    fn test_job_fail_message() {
        let mut job = make_job("job-2");
        job.fail("codec not found");
        assert_eq!(job.status, CloudJobStatus::Failed);
        assert_eq!(job.error_message.as_deref(), Some("codec not found"));
    }

    // 5. Job cancel
    #[test]
    fn test_job_cancel() {
        let mut job = make_job("job-3");
        job.cancel();
        assert_eq!(job.status, CloudJobStatus::Cancelled);
    }

    // 6. Built-in presets
    #[test]
    fn test_built_in_presets() {
        let hd = CloudTranscodePreset::hd_1080p();
        assert_eq!(hd.resolution, (1920, 1080));
        assert!(hd.has_video());

        let audio = CloudTranscodePreset::audio_only();
        assert_eq!(audio.resolution, (0, 0));
        assert!(!audio.has_video());
    }

    // 7. Preset pixel_count
    #[test]
    fn test_preset_pixel_count() {
        let p = CloudTranscodePreset::hd_720p();
        assert_eq!(p.pixel_count(), 1280 * 720);
    }

    // 8. Queue priority ordering
    #[test]
    fn test_queue_priority() {
        let mut q = CloudTranscodeQueue::new();
        let low = make_job("low");
        let high = make_job("high");
        q.submit(low, 1);
        q.submit(high, 200);
        // High-priority job should come out first
        let first = q.next().expect("first should be valid");
        assert_eq!(first.id, "high");
        let second = q.next().expect("second should be valid");
        assert_eq!(second.id, "low");
        assert!(q.is_empty());
    }

    // 9. Queue len / peek
    #[test]
    fn test_queue_len_and_peek() {
        let mut q = CloudTranscodeQueue::new();
        assert_eq!(q.len(), 0);
        q.submit(make_job("j1"), 10);
        q.submit(make_job("j2"), 20);
        assert_eq!(q.len(), 2);
        assert_eq!(q.peek().expect("peek should succeed").id, "j2");
    }

    // 10. Queue next on empty returns None
    #[test]
    fn test_queue_empty_next() {
        let mut q = CloudTranscodeQueue::new();
        assert!(q.next().is_none());
    }

    // 11. JobMetrics success_rate
    #[test]
    fn test_job_metrics() {
        let mut m = JobMetrics::new();
        m.record_submitted();
        m.record_submitted();
        m.record_completed(120.0);
        m.record_failed();
        assert_eq!(m.total_submitted, 2);
        assert!((m.success_rate() - 0.5).abs() < 1e-6);
    }

    // 12. CostEstimator HD cost
    #[test]
    fn test_cost_hd() {
        let est = CostEstimator::new();
        // 60 seconds → 1 minute at $0.024
        let cost = est.estimate_transcode_cost(60.0, (1920, 1080));
        assert!((cost - 0.024).abs() < 1e-9);
    }

    // 13. CostEstimator audio-only cost
    #[test]
    fn test_cost_audio_only() {
        let est = CostEstimator::new();
        let cost = est.estimate_transcode_cost(60.0, (0, 0));
        assert!((cost - 0.004).abs() < 1e-9);
    }

    // 14. CostEstimator job cost aggregates presets
    #[test]
    fn test_cost_job_aggregation() {
        let est = CostEstimator::new();
        let job = CloudTranscodeJob::new(
            "j",
            "s3://b/in.mp4",
            "s3://b/out/",
            vec!["hd_1080p".to_string(), "sd_480p".to_string()],
            60.0, // 1 minute source
        );
        let presets = vec![
            CloudTranscodePreset::hd_1080p(),
            CloudTranscodePreset::sd_480p(),
        ];
        let cost = est.estimate_job_cost(&job, &presets);
        // 1 min HD ($0.024) + 1 min SD ($0.017) = $0.041
        assert!((cost - 0.041).abs() < 1e-9);
    }
}
