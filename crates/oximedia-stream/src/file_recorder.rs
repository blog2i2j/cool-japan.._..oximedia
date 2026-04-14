//! File-path based live stream recorder for VOD capture.
//!
//! [`FileStreamRecorder`] writes raw segment data to numbered files in a
//! directory and tracks metadata for VOD reconstruction.  Unlike the
//! [`dvr_recorder`](crate::dvr_recorder) which manages a sliding look-back
//! window in memory, this recorder persists every byte to disk and is designed
//! for long-running live-to-VOD capture.
//!
//! # Usage
//!
//! ```rust,no_run
//! use oximedia_stream::file_recorder::FileStreamRecorder;
//!
//! let mut recorder = FileStreamRecorder::new("/tmp/capture");
//! recorder.write_segment(b"segment data here").expect("write");
//! recorder.write_segment(b"more segment data").expect("write");
//! let info = recorder.finalize().expect("finalize");
//! println!("recorded {} segments, {} bytes total", info.segment_count, info.total_bytes);
//! ```

use std::io;
use std::path::{Path, PathBuf};

// ─── RecorderInfo ─────────────────────────────────────────────────────────────

/// Summary information returned by [`FileStreamRecorder::finalize`].
#[derive(Debug, Clone)]
pub struct RecorderInfo {
    /// Directory where segment files were written.
    pub output_dir: PathBuf,
    /// Total number of segments written.
    pub segment_count: u64,
    /// Total bytes written across all segments.
    pub total_bytes: u64,
    /// Paths to all written segment files in write order.
    pub segment_paths: Vec<PathBuf>,
}

// ─── FileStreamRecorder ───────────────────────────────────────────────────────

/// Records live stream segments to numbered files on disk.
///
/// Each call to [`write_segment`](FileStreamRecorder::write_segment) creates a
/// new file named `{prefix}{sequence:010}.seg` in the configured output
/// directory.  [`finalize`](FileStreamRecorder::finalize) flushes any pending
/// state and returns a summary.
pub struct FileStreamRecorder {
    /// Directory where segment files are written.
    pub output_dir: PathBuf,
    /// Filename prefix, defaults to `"segment_"`.
    pub prefix: String,
    /// File extension, defaults to `"seg"`.
    pub extension: String,
    /// Monotonically increasing segment sequence number.
    segment_counter: u64,
    /// Total bytes written.
    total_bytes: u64,
    /// Paths written in order.
    segment_paths: Vec<PathBuf>,
    /// Whether `finalize` has been called.
    finalized: bool,
}

impl FileStreamRecorder {
    /// Create a new recorder that writes to `output_dir`.
    ///
    /// The directory is created if it does not exist when the first segment is
    /// written.
    pub fn new(output_dir: impl Into<PathBuf>) -> Self {
        Self {
            output_dir: output_dir.into(),
            prefix: "segment_".to_string(),
            extension: "seg".to_string(),
            segment_counter: 0,
            total_bytes: 0,
            segment_paths: Vec::new(),
            finalized: false,
        }
    }

    /// Override the filename prefix (default: `"segment_"`).
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// Override the file extension (default: `"seg"`).
    pub fn with_extension(mut self, ext: impl Into<String>) -> Self {
        self.extension = ext.into();
        self
    }

    /// Write `data` as the next segment file.
    ///
    /// Returns the path of the file that was written.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] if the output directory cannot be created or if
    /// the file write fails.
    pub fn write_segment(&mut self, data: &[u8]) -> io::Result<PathBuf> {
        if self.finalized {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "recorder has already been finalized",
            ));
        }
        // Ensure output directory exists.
        std::fs::create_dir_all(&self.output_dir)?;

        let path = self.segment_path(self.segment_counter);
        std::fs::write(&path, data)?;

        self.total_bytes = self.total_bytes.saturating_add(data.len() as u64);
        self.segment_paths.push(path.clone());
        self.segment_counter = self.segment_counter.saturating_add(1);
        Ok(path)
    }

    /// Finalize the recording and return a summary.
    ///
    /// After `finalize` any further calls to [`Self::write_segment`] will return an
    /// error.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] if the output directory does not exist (i.e. no
    /// segments have ever been written and the directory could not be verified).
    pub fn finalize(&mut self) -> io::Result<RecorderInfo> {
        self.finalized = true;
        // Verify the output directory exists (or is the empty case).
        if self.segment_counter == 0 {
            // Nothing was written — ensure the directory exists.
            std::fs::create_dir_all(&self.output_dir)?;
        }
        Ok(RecorderInfo {
            output_dir: self.output_dir.clone(),
            segment_count: self.segment_counter,
            total_bytes: self.total_bytes,
            segment_paths: self.segment_paths.clone(),
        })
    }

    /// Returns `true` if the recorder has been finalized.
    pub fn is_finalized(&self) -> bool {
        self.finalized
    }

    /// The number of segments written so far.
    pub fn segment_count(&self) -> u64 {
        self.segment_counter
    }

    /// Total bytes written so far.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Compute the path for a given sequence number.
    pub fn segment_path(&self, sequence: u64) -> PathBuf {
        self.output_dir.join(format!(
            "{}{:010}.{}",
            self.prefix, sequence, self.extension
        ))
    }

    /// Return a reference to the output directory path.
    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("oximedia_file_recorder_{tag}"))
    }

    #[test]
    fn test_write_single_segment() {
        let dir = tmp_dir("single");
        let _ = std::fs::remove_dir_all(&dir);
        let mut rec = FileStreamRecorder::new(&dir);
        let path = rec.write_segment(b"hello world").expect("write");
        assert!(path.exists());
        assert_eq!(std::fs::read(&path).expect("read"), b"hello world");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_segment_paths_numbered_sequentially() {
        let dir = tmp_dir("seq");
        let _ = std::fs::remove_dir_all(&dir);
        let mut rec = FileStreamRecorder::new(&dir);
        for i in 0..3u64 {
            rec.write_segment(&[i as u8]).expect("write");
        }
        assert_eq!(
            rec.segment_paths[0].file_name().and_then(|n| n.to_str()),
            Some("segment_0000000000.seg")
        );
        assert_eq!(
            rec.segment_paths[1].file_name().and_then(|n| n.to_str()),
            Some("segment_0000000001.seg")
        );
        assert_eq!(
            rec.segment_paths[2].file_name().and_then(|n| n.to_str()),
            Some("segment_0000000002.seg")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_finalize_returns_correct_summary() {
        let dir = tmp_dir("finalize");
        let _ = std::fs::remove_dir_all(&dir);
        let mut rec = FileStreamRecorder::new(&dir);
        rec.write_segment(b"aaa").expect("write");
        rec.write_segment(b"bb").expect("write");
        let info = rec.finalize().expect("finalize");
        assert_eq!(info.segment_count, 2);
        assert_eq!(info.total_bytes, 5);
        assert_eq!(info.segment_paths.len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_write_after_finalize_returns_error() {
        let dir = tmp_dir("post_finalize");
        let _ = std::fs::remove_dir_all(&dir);
        let mut rec = FileStreamRecorder::new(&dir);
        rec.finalize().expect("finalize");
        let err = rec.write_segment(b"should fail");
        assert!(err.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_custom_prefix_and_extension() {
        let dir = tmp_dir("custom_ext");
        let _ = std::fs::remove_dir_all(&dir);
        let mut rec = FileStreamRecorder::new(&dir)
            .with_prefix("chunk_")
            .with_extension("m4s");
        let path = rec.write_segment(b"fmp4").expect("write");
        let name = path.file_name().and_then(|n| n.to_str()).expect("name");
        assert!(name.starts_with("chunk_"));
        assert!(name.ends_with(".m4s"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_total_bytes_accumulate() {
        let dir = tmp_dir("bytes");
        let _ = std::fs::remove_dir_all(&dir);
        let mut rec = FileStreamRecorder::new(&dir);
        rec.write_segment(&[0u8; 100]).expect("write");
        rec.write_segment(&[0u8; 200]).expect("write");
        assert_eq!(rec.total_bytes(), 300);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_segment_count_reflects_writes() {
        let dir = tmp_dir("count");
        let _ = std::fs::remove_dir_all(&dir);
        let mut rec = FileStreamRecorder::new(&dir);
        assert_eq!(rec.segment_count(), 0);
        for i in 0..5u8 {
            rec.write_segment(&[i]).expect("write");
            assert_eq!(rec.segment_count(), (i + 1) as u64);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_is_finalized_flag() {
        let dir = tmp_dir("finalized_flag");
        let _ = std::fs::remove_dir_all(&dir);
        let mut rec = FileStreamRecorder::new(&dir);
        assert!(!rec.is_finalized());
        rec.write_segment(b"data").expect("write");
        assert!(!rec.is_finalized());
        rec.finalize().expect("finalize");
        assert!(rec.is_finalized());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_segment_path_naming_convention() {
        let dir = tmp_dir("naming");
        let rec = FileStreamRecorder::new(&dir);
        let path = rec.segment_path(42);
        let name = path.file_name().and_then(|n| n.to_str()).expect("name");
        assert_eq!(name, "segment_0000000042.seg");
    }
}
