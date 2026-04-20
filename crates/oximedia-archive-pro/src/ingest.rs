//! Archive ingest pipeline module.
//!
//! Provides a structured pipeline for submitting, processing and tracking
//! media assets as they move through the archive ingest workflow.

use serde::{Deserialize, Serialize};

/// The tier an asset has reached in the ingest lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IngestTier {
    /// Raw asset has been received but not yet examined
    Raw,
    /// Asset is undergoing format and integrity checks
    Processed,
    /// All checks have passed; asset awaits archival
    Verified,
    /// Asset has been written to the long-term archive
    Archived,
}

/// A media asset candidate waiting to enter the archive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestCandidate {
    /// Filesystem path to the asset
    pub path: String,
    /// File size in bytes
    pub size_bytes: u64,
    /// MIME type or format identifier (e.g. `"video/x-matroska"`)
    pub media_type: String,
    /// Pre-computed checksum (optional; may be populated during processing)
    pub checksum: Option<String>,
}

impl IngestCandidate {
    /// Create a new `IngestCandidate` without a pre-computed checksum.
    #[must_use]
    pub fn new(path: &str, size: u64, media_type: &str) -> Self {
        Self {
            path: path.to_owned(),
            size_bytes: size,
            media_type: media_type.to_owned(),
            checksum: None,
        }
    }

    /// Rough estimate of wall-clock processing time in seconds.
    ///
    /// Assumes a throughput of 100 MB/s and a minimum of 1 second.
    #[must_use]
    pub fn estimated_processing_time_s(&self) -> u64 {
        const THROUGHPUT_BYTES_PER_SEC: u64 = 100 * 1024 * 1024; // 100 MB/s
        let secs = self.size_bytes / THROUGHPUT_BYTES_PER_SEC;
        secs.max(1)
    }
}

/// A staged pipeline that manages assets through ingest phases.
#[derive(Debug, Default)]
pub struct IngestPipeline {
    /// Assets waiting to be processed
    pub pending: Vec<IngestCandidate>,
    /// Assets currently being processed
    pub in_progress: Vec<IngestCandidate>,
    /// Paths of successfully archived assets
    pub completed: Vec<String>,
    /// Paths of failed assets and their error reasons
    pub failed: Vec<(String, String)>,
}

impl IngestPipeline {
    /// Create a new empty `IngestPipeline`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Submit a new candidate to the pending queue.
    pub fn submit(&mut self, candidate: IngestCandidate) {
        self.pending.push(candidate);
    }

    /// Move the next pending asset into the in-progress list and return it.
    ///
    /// Returns `None` if the pending queue is empty.
    pub fn start_next(&mut self) -> Option<IngestCandidate> {
        if self.pending.is_empty() {
            return None;
        }
        let candidate = self.pending.remove(0);
        self.in_progress.push(candidate.clone());
        Some(candidate)
    }

    /// Mark an in-progress asset as successfully completed.
    ///
    /// Moves it from the in-progress list to the completed list.
    pub fn complete(&mut self, path: &str) {
        if let Some(pos) = self.in_progress.iter().position(|c| c.path == path) {
            self.in_progress.remove(pos);
        }
        self.completed.push(path.to_owned());
    }

    /// Mark an in-progress asset as failed.
    ///
    /// Moves it from the in-progress list to the failed list with a reason.
    pub fn fail(&mut self, path: &str, reason: &str) {
        if let Some(pos) = self.in_progress.iter().position(|c| c.path == path) {
            self.in_progress.remove(pos);
        }
        self.failed.push((path.to_owned(), reason.to_owned()));
    }

    /// Return a snapshot of current pipeline statistics.
    #[must_use]
    pub fn stats(&self) -> IngestStats {
        let total_bytes_processed: u64 = self.in_progress.iter().map(|c| c.size_bytes).sum::<u64>();

        IngestStats {
            pending: self.pending.len(),
            in_progress: self.in_progress.len(),
            completed: self.completed.len(),
            failed: self.failed.len(),
            total_bytes_processed,
        }
    }
}

/// Snapshot statistics for an `IngestPipeline`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestStats {
    /// Number of assets waiting to be processed
    pub pending: usize,
    /// Number of assets currently being processed
    pub in_progress: usize,
    /// Number of assets that completed successfully
    pub completed: usize,
    /// Number of assets that failed
    pub failed: usize,
    /// Aggregate bytes of in-progress assets
    pub total_bytes_processed: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(path: &str, size: u64) -> IngestCandidate {
        IngestCandidate::new(path, size, "video/x-matroska")
    }

    fn tmp_str(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("oximedia-archive-pro-ingest-{name}"))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn test_ingest_candidate_new() {
        let path = tmp_str("a.mkv");
        let c = candidate(&path, 1024);
        assert_eq!(c.path, path);
        assert_eq!(c.size_bytes, 1024);
        assert!(c.checksum.is_none());
    }

    #[test]
    fn test_estimated_processing_time_minimum() {
        // File smaller than 100 MB → at least 1 second
        let c = candidate(&tmp_str("small.mkv"), 1024);
        assert_eq!(c.estimated_processing_time_s(), 1);
    }

    #[test]
    fn test_estimated_processing_time_large() {
        // 200 MB file → ~2 seconds
        let c = candidate(&tmp_str("large.mkv"), 200 * 1024 * 1024);
        assert_eq!(c.estimated_processing_time_s(), 2);
    }

    #[test]
    fn test_ingest_tier_variants() {
        // Ensure all variants can be constructed
        let _r = IngestTier::Raw;
        let _p = IngestTier::Processed;
        let _v = IngestTier::Verified;
        let _a = IngestTier::Archived;
    }

    #[test]
    fn test_pipeline_submit_and_stats() {
        let mut p = IngestPipeline::new();
        p.submit(candidate("/a.mkv", 512));
        p.submit(candidate("/b.mkv", 1024));
        let stats = p.stats();
        assert_eq!(stats.pending, 2);
        assert_eq!(stats.in_progress, 0);
    }

    #[test]
    fn test_pipeline_start_next_fifo() {
        let mut p = IngestPipeline::new();
        p.submit(candidate("/first.mkv", 100));
        p.submit(candidate("/second.mkv", 200));
        let c = p.start_next().expect("operation should succeed");
        assert_eq!(c.path, "/first.mkv");
        assert_eq!(p.pending.len(), 1);
        assert_eq!(p.in_progress.len(), 1);
    }

    #[test]
    fn test_pipeline_start_next_empty() {
        let mut p = IngestPipeline::new();
        assert!(p.start_next().is_none());
    }

    #[test]
    fn test_pipeline_complete() {
        let mut p = IngestPipeline::new();
        p.submit(candidate("/a.mkv", 100));
        p.start_next();
        p.complete("/a.mkv");
        let stats = p.stats();
        assert_eq!(stats.in_progress, 0);
        assert_eq!(stats.completed, 1);
    }

    #[test]
    fn test_pipeline_fail() {
        let mut p = IngestPipeline::new();
        p.submit(candidate("/bad.mkv", 100));
        p.start_next();
        p.fail("/bad.mkv", "checksum mismatch");
        let stats = p.stats();
        assert_eq!(stats.in_progress, 0);
        assert_eq!(stats.failed, 1);
        assert_eq!(p.failed[0].1, "checksum mismatch");
    }

    #[test]
    fn test_pipeline_multiple_operations() {
        let mut p = IngestPipeline::new();
        for i in 0..5 {
            p.submit(candidate(&format!("/f{i}.mkv"), 1024));
        }
        p.start_next();
        p.start_next();
        p.complete("/f0.mkv");
        p.fail("/f1.mkv", "corrupt");
        let stats = p.stats();
        assert_eq!(stats.pending, 3);
        assert_eq!(stats.in_progress, 0);
        assert_eq!(stats.completed, 1);
        assert_eq!(stats.failed, 1);
    }

    #[test]
    fn test_stats_total_bytes_in_progress() {
        let mut p = IngestPipeline::new();
        p.submit(candidate("/a.mkv", 500));
        p.submit(candidate("/b.mkv", 300));
        p.start_next();
        p.start_next();
        let stats = p.stats();
        assert_eq!(stats.total_bytes_processed, 800);
    }

    #[test]
    fn test_complete_unknown_path_is_noop() {
        let mut p = IngestPipeline::new();
        // Should not panic
        p.complete("/nonexistent.mkv");
        assert_eq!(p.completed.len(), 1); // still added to completed list
    }

    #[test]
    fn test_fail_unknown_path_is_noop() {
        let mut p = IngestPipeline::new();
        p.fail("/nonexistent.mkv", "oops");
        assert_eq!(p.failed.len(), 1);
    }
}
