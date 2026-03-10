#![allow(dead_code)]
//! Chunked upload management for cloud storage.

/// State of an upload job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UploadState {
    /// Upload has not yet started.
    Pending,
    /// Upload is currently in progress.
    Uploading,
    /// Upload has been paused.
    Paused,
    /// Upload completed successfully.
    Completed,
    /// Upload failed with an error message.
    Failed(String),
}

impl UploadState {
    /// Returns true if the upload is actively in progress.
    pub fn is_active(&self) -> bool {
        matches!(self, UploadState::Uploading)
    }

    /// Returns true if the upload has finished (either completed or failed).
    pub fn is_terminal(&self) -> bool {
        matches!(self, UploadState::Completed | UploadState::Failed(_))
    }

    /// Returns true if the upload completed successfully.
    pub fn is_success(&self) -> bool {
        matches!(self, UploadState::Completed)
    }
}

/// A single chunk of a multipart upload.
#[derive(Debug, Clone)]
pub struct UploadChunk {
    /// Zero-based index of this chunk.
    pub index: usize,
    /// Byte offset within the full file.
    pub offset: u64,
    /// Size of this chunk in bytes.
    pub size: u64,
    /// Whether this chunk has been successfully uploaded.
    pub uploaded: bool,
    /// ETag or checksum returned by the cloud provider.
    pub etag: Option<String>,
}

impl UploadChunk {
    /// Creates a new chunk descriptor.
    pub fn new(index: usize, offset: u64, size: u64) -> Self {
        Self {
            index,
            offset,
            size,
            uploaded: false,
            etag: None,
        }
    }

    /// Marks the chunk as uploaded with the given ETag.
    pub fn mark_uploaded(&mut self, etag: impl Into<String>) {
        self.uploaded = true;
        self.etag = Some(etag.into());
    }

    /// Returns true if this chunk has been successfully uploaded.
    pub fn is_complete(&self) -> bool {
        self.uploaded
    }

    /// Returns the end byte offset (exclusive) of this chunk.
    pub fn end_offset(&self) -> u64 {
        self.offset + self.size
    }
}

/// An upload job tracking the full multipart upload of a single file.
#[derive(Debug)]
pub struct UploadJob {
    /// Unique job identifier.
    pub id: u64,
    /// Remote destination key or path.
    pub destination: String,
    /// Total size of the file in bytes.
    pub total_size: u64,
    /// Current state of the job.
    pub state: UploadState,
    /// Individual chunks.
    chunks: Vec<UploadChunk>,
}

impl UploadJob {
    /// Creates a new upload job and automatically splits the file into chunks.
    pub fn new(id: u64, destination: impl Into<String>, total_size: u64, chunk_size: u64) -> Self {
        let dest = destination.into();
        let chunk_size = chunk_size.max(1);
        let num_chunks = total_size.div_ceil(chunk_size) as usize;
        let chunks = (0..num_chunks)
            .map(|i| {
                let offset = i as u64 * chunk_size;
                let size = chunk_size.min(total_size - offset);
                UploadChunk::new(i, offset, size)
            })
            .collect();
        Self {
            id,
            destination: dest,
            total_size,
            state: UploadState::Pending,
            chunks,
        }
    }

    /// Returns the total number of chunks in this job.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Returns the number of chunks that have been uploaded.
    pub fn uploaded_chunk_count(&self) -> usize {
        self.chunks.iter().filter(|c| c.is_complete()).count()
    }

    /// Returns the upload progress as a percentage in `[0.0, 100.0]`.
    #[allow(clippy::cast_precision_loss)]
    pub fn progress_pct(&self) -> f64 {
        if self.chunks.is_empty() {
            return 100.0;
        }
        let done = self.uploaded_chunk_count() as f64;
        let total = self.chunks.len() as f64;
        (done / total) * 100.0
    }

    /// Marks a chunk by index as uploaded.
    ///
    /// Returns `false` if the index is out of bounds.
    pub fn complete_chunk(&mut self, index: usize, etag: impl Into<String>) -> bool {
        if let Some(chunk) = self.chunks.get_mut(index) {
            chunk.mark_uploaded(etag);
            true
        } else {
            false
        }
    }

    /// Returns an iterator over all chunks.
    pub fn chunks(&self) -> impl Iterator<Item = &UploadChunk> {
        self.chunks.iter()
    }

    /// Returns the number of bytes uploaded so far.
    pub fn uploaded_bytes(&self) -> u64 {
        self.chunks
            .iter()
            .filter(|c| c.is_complete())
            .map(|c| c.size)
            .sum()
    }
}

/// Manager that tracks multiple concurrent upload jobs.
#[derive(Debug, Default)]
pub struct UploadManager {
    jobs: Vec<UploadJob>,
    next_id: u64,
    /// Default chunk size in bytes (5 MiB).
    chunk_size: u64,
}

impl UploadManager {
    /// Creates a new upload manager with the default 5 MiB chunk size.
    pub fn new() -> Self {
        Self {
            jobs: Vec::new(),
            next_id: 0,
            chunk_size: 5 * 1024 * 1024,
        }
    }

    /// Creates a new upload manager with a custom chunk size.
    pub fn with_chunk_size(chunk_size: u64) -> Self {
        Self {
            jobs: Vec::new(),
            next_id: 0,
            chunk_size,
        }
    }

    /// Creates and registers a new upload job.
    pub fn create_job(&mut self, destination: impl Into<String>, total_size: u64) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let job = UploadJob::new(id, destination, total_size, self.chunk_size);
        self.jobs.push(job);
        id
    }

    /// Returns a reference to a job by its ID.
    pub fn get_job(&self, id: u64) -> Option<&UploadJob> {
        self.jobs.iter().find(|j| j.id == id)
    }

    /// Returns a mutable reference to a job by its ID.
    pub fn get_job_mut(&mut self, id: u64) -> Option<&mut UploadJob> {
        self.jobs.iter_mut().find(|j| j.id == id)
    }

    /// Returns the number of jobs currently managed.
    pub fn job_count(&self) -> usize {
        self.jobs.len()
    }

    /// Returns the number of active (uploading) jobs.
    pub fn active_job_count(&self) -> usize {
        self.jobs.iter().filter(|j| j.state.is_active()).count()
    }

    /// Returns the chunk count for a given job, or 0 if not found.
    pub fn chunk_count(&self, job_id: u64) -> usize {
        self.get_job(job_id).map_or(0, |j| j.chunk_count())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upload_state_is_active() {
        assert!(UploadState::Uploading.is_active());
        assert!(!UploadState::Pending.is_active());
        assert!(!UploadState::Completed.is_active());
    }

    #[test]
    fn test_upload_state_is_terminal() {
        assert!(UploadState::Completed.is_terminal());
        assert!(UploadState::Failed("err".into()).is_terminal());
        assert!(!UploadState::Uploading.is_terminal());
    }

    #[test]
    fn test_upload_state_is_success() {
        assert!(UploadState::Completed.is_success());
        assert!(!UploadState::Failed("x".into()).is_success());
    }

    #[test]
    fn test_chunk_is_complete_initially_false() {
        let c = UploadChunk::new(0, 0, 1024);
        assert!(!c.is_complete());
    }

    #[test]
    fn test_chunk_mark_uploaded() {
        let mut c = UploadChunk::new(0, 0, 1024);
        c.mark_uploaded("etag-abc");
        assert!(c.is_complete());
        assert_eq!(c.etag.as_deref(), Some("etag-abc"));
    }

    #[test]
    fn test_chunk_end_offset() {
        let c = UploadChunk::new(0, 1000, 500);
        assert_eq!(c.end_offset(), 1500);
    }

    #[test]
    fn test_upload_job_chunk_count() {
        // 10 MiB file, 5 MiB chunks → 2 chunks
        let job = UploadJob::new(0, "dest/file.mp4", 10 * 1024 * 1024, 5 * 1024 * 1024);
        assert_eq!(job.chunk_count(), 2);
    }

    #[test]
    fn test_upload_job_chunk_count_remainder() {
        // 11 MiB file, 5 MiB chunks → 3 chunks
        let job = UploadJob::new(0, "dest/file.mp4", 11 * 1024 * 1024, 5 * 1024 * 1024);
        assert_eq!(job.chunk_count(), 3);
    }

    #[test]
    fn test_progress_pct_initial() {
        let job = UploadJob::new(0, "dest", 1000, 100);
        assert!((job.progress_pct() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_progress_pct_after_completion() {
        let mut job = UploadJob::new(0, "dest", 1000, 100);
        for i in 0..10 {
            job.complete_chunk(i, format!("etag-{}", i));
        }
        assert!((job.progress_pct() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_progress_pct_partial() {
        let mut job = UploadJob::new(0, "dest", 1000, 100);
        job.complete_chunk(0, "e0");
        job.complete_chunk(1, "e1");
        job.complete_chunk(2, "e2");
        job.complete_chunk(3, "e3");
        job.complete_chunk(4, "e4");
        // 5 of 10 done
        assert!((job.progress_pct() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_complete_chunk_out_of_bounds() {
        let mut job = UploadJob::new(0, "dest", 100, 100);
        assert!(!job.complete_chunk(999, "etag"));
    }

    #[test]
    fn test_uploaded_bytes() {
        let mut job = UploadJob::new(0, "dest", 300, 100);
        job.complete_chunk(0, "e0");
        assert_eq!(job.uploaded_bytes(), 100);
    }

    #[test]
    fn test_manager_create_job() {
        let mut mgr = UploadManager::with_chunk_size(100);
        let id = mgr.create_job("remote/file.mp4", 300);
        assert_eq!(mgr.job_count(), 1);
        assert_eq!(mgr.chunk_count(id), 3);
    }

    #[test]
    fn test_manager_get_job() {
        let mut mgr = UploadManager::with_chunk_size(100);
        let id = mgr.create_job("file.mp4", 200);
        let job = mgr.get_job(id);
        assert!(job.is_some());
    }

    #[test]
    fn test_manager_active_job_count() {
        let mut mgr = UploadManager::with_chunk_size(100);
        let id = mgr.create_job("file.mp4", 100);
        mgr.get_job_mut(id)
            .expect("get_job_mut should succeed")
            .state = UploadState::Uploading;
        assert_eq!(mgr.active_job_count(), 1);
    }
}
