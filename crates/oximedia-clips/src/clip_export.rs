//! Clip export job management for queuing and tracking export operations.
//!
//! Provides `ExportTarget`, `ClipExportConfig`, `ClipExportJob`, and `ClipExporter`.

#![allow(dead_code)]

/// The target format for a clip export.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportTarget {
    /// Export as an MP4 file.
    Mp4,
    /// Export as a ProRes MOV file.
    ProRes,
    /// Export as a DNxHD MXF file.
    DnxHd,
    /// Export as a still image sequence (PNG).
    ImageSequence,
    /// Export as an AAF interchange file.
    Aaf,
    /// Export as an XML interchange file (e.g. Final Cut).
    Xml,
}

impl ExportTarget {
    /// Returns the common file format name for this target.
    pub fn format_name(&self) -> &'static str {
        match self {
            ExportTarget::Mp4 => "MP4 (H.264/HEVC)",
            ExportTarget::ProRes => "Apple ProRes MOV",
            ExportTarget::DnxHd => "Avid DNxHD MXF",
            ExportTarget::ImageSequence => "PNG Image Sequence",
            ExportTarget::Aaf => "AAF Interchange",
            ExportTarget::Xml => "XML Interchange",
        }
    }

    /// File extension typically used for this target.
    pub fn extension(&self) -> &'static str {
        match self {
            ExportTarget::Mp4 => "mp4",
            ExportTarget::ProRes => "mov",
            ExportTarget::DnxHd => "mxf",
            ExportTarget::ImageSequence => "png",
            ExportTarget::Aaf => "aaf",
            ExportTarget::Xml => "xml",
        }
    }

    /// Returns `true` for lossless or near-lossless professional formats.
    pub fn is_professional_format(&self) -> bool {
        matches!(
            self,
            ExportTarget::ProRes | ExportTarget::DnxHd | ExportTarget::Aaf
        )
    }
}

/// Configuration for a single clip export operation.
#[derive(Debug, Clone)]
pub struct ClipExportConfig {
    /// The source clip identifier.
    pub clip_id: String,
    /// Destination directory path.
    pub output_dir: String,
    /// Target format.
    pub target: ExportTarget,
    /// Output video width in pixels (0 = source).
    pub width: u32,
    /// Output video height in pixels (0 = source).
    pub height: u32,
    /// Target bitrate in kbps (0 = auto).
    pub bitrate_kbps: u32,
    /// Whether to include embedded metadata.
    pub include_metadata: bool,
}

impl ClipExportConfig {
    /// Create a minimal export config for the given clip and target.
    pub fn new(
        clip_id: impl Into<String>,
        output_dir: impl Into<String>,
        target: ExportTarget,
    ) -> Self {
        Self {
            clip_id: clip_id.into(),
            output_dir: output_dir.into(),
            target,
            width: 0,
            height: 0,
            bitrate_kbps: 0,
            include_metadata: true,
        }
    }

    /// Set output resolution.
    pub fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set target bitrate in kbps.
    pub fn with_bitrate(mut self, kbps: u32) -> Self {
        self.bitrate_kbps = kbps;
        self
    }

    /// Returns `true` when the config has the minimum required fields set.
    pub fn is_valid(&self) -> bool {
        !self.clip_id.is_empty() && !self.output_dir.is_empty()
    }
}

/// Status of an export job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportJobStatus {
    /// Waiting in queue.
    Pending,
    /// Currently being processed.
    Running,
    /// Successfully completed.
    Completed,
    /// Failed with an error message.
    Failed(String),
}

impl ExportJobStatus {
    /// Returns `true` if the job has finished (successfully or not).
    pub fn is_done(&self) -> bool {
        matches!(
            self,
            ExportJobStatus::Completed | ExportJobStatus::Failed(_)
        )
    }
}

/// A queued export job with its configuration and current status.
#[derive(Debug, Clone)]
pub struct ClipExportJob {
    /// Unique job identifier.
    pub job_id: String,
    /// Export configuration.
    pub config: ClipExportConfig,
    /// Current job status.
    pub status: ExportJobStatus,
    /// Duration of the clip in seconds (used for size estimation).
    pub duration_secs: f64,
}

impl ClipExportJob {
    /// Create a new pending export job.
    pub fn new(job_id: impl Into<String>, config: ClipExportConfig, duration_secs: f64) -> Self {
        Self {
            job_id: job_id.into(),
            config,
            status: ExportJobStatus::Pending,
            duration_secs,
        }
    }

    /// Rough estimate of output file size in megabytes.
    ///
    /// Uses bitrate when set, otherwise applies a format-specific heuristic.
    #[allow(clippy::cast_precision_loss)]
    pub fn estimated_size_mb(&self) -> f64 {
        let effective_kbps = if self.config.bitrate_kbps > 0 {
            self.config.bitrate_kbps as f64
        } else {
            match &self.config.target {
                ExportTarget::Mp4 => 8_000.0,
                ExportTarget::ProRes => 145_000.0,
                ExportTarget::DnxHd => 120_000.0,
                ExportTarget::ImageSequence => 4_000.0,
                ExportTarget::Aaf | ExportTarget::Xml => 500.0,
            }
        };
        // size_mb = bitrate_kbps * duration_secs / 8 / 1024
        effective_kbps * self.duration_secs / 8.0 / 1024.0
    }

    /// Mark this job as running.
    pub fn start(&mut self) {
        self.status = ExportJobStatus::Running;
    }

    /// Mark this job as completed.
    pub fn complete(&mut self) {
        self.status = ExportJobStatus::Completed;
    }

    /// Mark this job as failed with a reason.
    pub fn fail(&mut self, reason: impl Into<String>) {
        self.status = ExportJobStatus::Failed(reason.into());
    }
}

/// Manages a queue of `ClipExportJob`s.
#[derive(Debug, Default)]
pub struct ClipExporter {
    jobs: Vec<ClipExportJob>,
    next_id: u64,
}

impl ClipExporter {
    /// Create a new, empty exporter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an export config to the queue. Returns the assigned job id.
    pub fn queue(&mut self, config: ClipExportConfig, duration_secs: f64) -> String {
        let job_id = format!("job-{}", self.next_id);
        self.next_id += 1;
        self.jobs
            .push(ClipExportJob::new(job_id.clone(), config, duration_secs));
        job_id
    }

    /// Number of jobs currently in `Pending` state.
    pub fn pending_count(&self) -> usize {
        self.jobs
            .iter()
            .filter(|j| j.status == ExportJobStatus::Pending)
            .count()
    }

    /// Total number of jobs (all statuses).
    pub fn total_count(&self) -> usize {
        self.jobs.len()
    }

    /// Find a job by its id.
    pub fn get_job(&self, job_id: &str) -> Option<&ClipExportJob> {
        self.jobs.iter().find(|j| j.job_id == job_id)
    }

    /// Find a job mutably by its id.
    pub fn get_job_mut(&mut self, job_id: &str) -> Option<&mut ClipExportJob> {
        self.jobs.iter_mut().find(|j| j.job_id == job_id)
    }

    /// Returns all jobs with `Pending` status.
    pub fn pending_jobs(&self) -> Vec<&ClipExportJob> {
        self.jobs
            .iter()
            .filter(|j| j.status == ExportJobStatus::Pending)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_str(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("oximedia-clips-export-{name}"))
            .to_string_lossy()
            .into_owned()
    }

    fn mp4_config(clip_id: &str) -> ClipExportConfig {
        ClipExportConfig::new(clip_id, tmp_str("export"), ExportTarget::Mp4)
    }

    #[test]
    fn export_target_format_name_mp4() {
        assert_eq!(ExportTarget::Mp4.format_name(), "MP4 (H.264/HEVC)");
    }

    #[test]
    fn export_target_extension() {
        assert_eq!(ExportTarget::ProRes.extension(), "mov");
        assert_eq!(ExportTarget::DnxHd.extension(), "mxf");
    }

    #[test]
    fn export_target_is_professional() {
        assert!(ExportTarget::ProRes.is_professional_format());
        assert!(!ExportTarget::Mp4.is_professional_format());
    }

    #[test]
    fn export_config_is_valid_with_fields() {
        let cfg = mp4_config("clip-1");
        assert!(cfg.is_valid());
    }

    #[test]
    fn export_config_invalid_when_empty_clip_id() {
        let cfg = ClipExportConfig::new("", "/tmp", ExportTarget::Mp4);
        assert!(!cfg.is_valid());
    }

    #[test]
    fn export_config_invalid_when_empty_output_dir() {
        let cfg = ClipExportConfig::new("c1", "", ExportTarget::Mp4);
        assert!(!cfg.is_valid());
    }

    #[test]
    fn export_job_initial_status_pending() {
        let job = ClipExportJob::new("j1", mp4_config("c1"), 60.0);
        assert_eq!(job.status, ExportJobStatus::Pending);
    }

    #[test]
    fn export_job_estimated_size_uses_bitrate() {
        let cfg = mp4_config("c1").with_bitrate(10_000); // 10 Mbps
        let job = ClipExportJob::new("j1", cfg, 10.0); // 10 seconds
        let size = job.estimated_size_mb();
        // 10_000 * 10 / 8 / 1024 ≈ 12.2 MB
        assert!(size > 10.0 && size < 15.0);
    }

    #[test]
    fn export_job_estimated_size_fallback_prores() {
        let cfg = ClipExportConfig::new("c1", "/tmp", ExportTarget::ProRes);
        let job = ClipExportJob::new("j1", cfg, 60.0);
        // ProRes at 145 Mbps for 60 sec should be large
        assert!(job.estimated_size_mb() > 100.0);
    }

    #[test]
    fn export_job_status_transitions() {
        let mut job = ClipExportJob::new("j1", mp4_config("c1"), 30.0);
        job.start();
        assert_eq!(job.status, ExportJobStatus::Running);
        job.complete();
        assert_eq!(job.status, ExportJobStatus::Completed);
        assert!(job.status.is_done());
    }

    #[test]
    fn export_job_fail_status() {
        let mut job = ClipExportJob::new("j1", mp4_config("c1"), 30.0);
        job.fail("codec not found");
        assert!(matches!(job.status, ExportJobStatus::Failed(_)));
        assert!(job.status.is_done());
    }

    #[test]
    fn exporter_queue_increments_pending() {
        let mut exporter = ClipExporter::new();
        assert_eq!(exporter.pending_count(), 0);
        exporter.queue(mp4_config("c1"), 60.0);
        exporter.queue(mp4_config("c2"), 30.0);
        assert_eq!(exporter.pending_count(), 2);
    }

    #[test]
    fn exporter_queue_returns_job_id() {
        let mut exporter = ClipExporter::new();
        let id = exporter.queue(mp4_config("c1"), 10.0);
        assert_eq!(id, "job-0");
        let id2 = exporter.queue(mp4_config("c2"), 10.0);
        assert_eq!(id2, "job-1");
    }

    #[test]
    fn exporter_get_job_found() {
        let mut exporter = ClipExporter::new();
        let id = exporter.queue(mp4_config("c1"), 10.0);
        assert!(exporter.get_job(&id).is_some());
    }

    #[test]
    fn exporter_pending_count_decreases_after_completion() {
        let mut exporter = ClipExporter::new();
        let id = exporter.queue(mp4_config("c1"), 10.0);
        if let Some(job) = exporter.get_job_mut(&id) {
            job.complete();
        }
        assert_eq!(exporter.pending_count(), 0);
    }
}
