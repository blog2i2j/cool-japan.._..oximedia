//! Multi-file batch metadata update.
//!
//! [`BatchMetadataUpdate`] applies a uniform set of tag changes to a list of
//! media files in a single pass.  Results are collected into a [`BatchResult`]
//! that separates successfully updated paths from failures, without aborting on
//! the first error.
//!
//! Internally the implementation delegates to the single-file
//! [`super::editor::BatchMetadataEditor`] for each path, so the same
//! format-detection and round-trip write logic is reused.
//!
//! # Example
//!
//! ```ignore
//! use oximedia_container::metadata::batch::{BatchMetadataUpdate, BatchResult};
//!
//! let result = BatchMetadataUpdate::new()
//!     .add_file("track01.flac")
//!     .add_file("track02.flac")
//!     .set_tag("ALBUM", "Greatest Hits")
//!     .set_tag("DATE", "2026")
//!     .apply();
//!
//! println!("{}", result.into_report());
//! ```

use std::path::{Path, PathBuf};

// ─── BatchError ───────────────────────────────────────────────────────────────

/// An error that occurred while processing a single file in the batch.
#[derive(Debug)]
pub struct BatchError(pub String);

impl std::fmt::Display for BatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for BatchError {}

impl BatchError {
    fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

// ─── BatchResult ──────────────────────────────────────────────────────────────

/// Outcome of a [`BatchMetadataUpdate::apply`] call.
///
/// Paths that were updated successfully are in [`ok`]; paths where the update
/// failed are in [`failed`] together with the associated error.
///
/// [`ok`]: BatchResult::ok
/// [`failed`]: BatchResult::failed
#[derive(Debug, Default)]
pub struct BatchResult {
    /// Paths that were updated without error.
    pub ok: Vec<PathBuf>,
    /// Paths where the update failed, paired with the error.
    pub failed: Vec<(PathBuf, BatchError)>,
}

impl BatchResult {
    /// Formats a human-readable one-line summary.
    #[must_use]
    pub fn into_report(self) -> String {
        let ok_count = self.ok.len();
        if self.failed.is_empty() {
            format!("Updated {} file(s) successfully.", ok_count)
        } else {
            let details: Vec<String> = self
                .failed
                .iter()
                .map(|(p, e)| format!("{}: {}", p.display(), e))
                .collect();
            format!(
                "Updated {} file(s). Failed on {} file(s): {}",
                ok_count,
                self.failed.len(),
                details.join("; ")
            )
        }
    }

    /// Returns `true` if every file was updated without error.
    #[must_use]
    pub fn all_succeeded(&self) -> bool {
        self.failed.is_empty()
    }

    /// Returns the total number of files processed (succeeded + failed).
    #[must_use]
    pub fn total(&self) -> usize {
        self.ok.len() + self.failed.len()
    }
}

// ─── BatchMetadataUpdate ──────────────────────────────────────────────────────

/// Builder for applying a uniform set of metadata tag changes to multiple files.
///
/// Chain builder calls to configure the target files and tags, then call
/// [`apply`] to execute the batch.  One error in a single file does not abort
/// the remaining files; all results are collected and returned as a [`BatchResult`].
///
/// [`apply`]: BatchMetadataUpdate::apply
#[derive(Debug, Default)]
pub struct BatchMetadataUpdate {
    /// Target file paths.
    files: Vec<PathBuf>,
    /// (key, value) pairs to write on every target file.
    tags: Vec<(String, String)>,
    /// Optional source file and list of tag keys to copy from it.
    ///
    /// Tags copied from the source are merged with `tags`; entries in `tags`
    /// take precedence over those copied from the source.
    copy_from: Option<(PathBuf, Vec<String>)>,
}

impl BatchMetadataUpdate {
    /// Creates a new, empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a target file path.
    #[must_use]
    pub fn add_file(mut self, path: impl AsRef<Path>) -> Self {
        self.files.push(path.as_ref().to_path_buf());
        self
    }

    /// Schedules a tag `key` → `value` write for every target file.
    ///
    /// If the same key is added more than once, the last value wins.
    #[must_use]
    pub fn set_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.push((key.into(), value.into()));
        self
    }

    /// Configures the update to also copy specific tags from `source` into
    /// each target file before applying the explicit [`set_tag`] entries.
    ///
    /// When a key appears in both `source` and the explicit tags, the
    /// explicit value wins.
    ///
    /// [`set_tag`]: BatchMetadataUpdate::set_tag
    #[must_use]
    pub fn copy_from(mut self, source: impl AsRef<Path>, keys: Vec<String>) -> Self {
        self.copy_from = Some((source.as_ref().to_path_buf(), keys));
        self
    }

    /// Executes the batch update and returns a [`BatchResult`].
    ///
    /// Each file is processed independently; a failure on one file does not
    /// abort the remaining files.
    pub fn apply(self) -> BatchResult {
        let mut result = BatchResult::default();

        // Pre-compute merged tag list (copy_from source tags + explicit tags).
        let merged_tags: Vec<(String, String)> = match self.build_merged_tags() {
            Ok(tags) => tags,
            Err(e) => {
                // If we cannot read the copy_from source we fail every file.
                let msg = format!("failed to read copy_from source: {e}");
                for path in self.files {
                    result.failed.push((path, BatchError::new(msg.clone())));
                }
                return result;
            }
        };

        for path in &self.files {
            match apply_tags_to_file(path, &merged_tags) {
                Ok(()) => result.ok.push(path.clone()),
                Err(e) => result.failed.push((path.clone(), e)),
            }
        }

        result
    }

    /// Builds the final `(key, value)` list from `copy_from` + explicit tags.
    ///
    /// The copy_from source entries come first; explicit entries follow, so
    /// explicit values override copied ones when processed by the editor.
    fn build_merged_tags(&self) -> Result<Vec<(String, String)>, BatchError> {
        let mut merged: Vec<(String, String)> = Vec::new();

        if let Some((source_path, keys)) = &self.copy_from {
            let source_tags = read_tags_from_file(source_path, keys)?;
            merged.extend(source_tags);
        }

        // Explicit tags always override copy_from values.
        merged.extend(self.tags.iter().cloned());

        Ok(merged)
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Reads the specified tag keys from `path` and returns (key, text_value) pairs.
///
/// Only `Text` tag values are returned; binary tags are silently skipped.
///
/// On WASM there is no async file I/O available so this returns an error.
#[cfg(target_arch = "wasm32")]
fn read_tags_from_file(
    _path: &Path,
    _keys: &[String],
) -> Result<Vec<(String, String)>, BatchError> {
    Err(BatchError::new("file I/O is not supported on WASM"))
}

#[cfg(not(target_arch = "wasm32"))]
fn read_tags_from_file(path: &Path, keys: &[String]) -> Result<Vec<(String, String)>, BatchError> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| BatchError::new(e.to_string()))?;

    rt.block_on(async {
        use crate::metadata::editor::MetadataEditor;

        let editor = MetadataEditor::open(path)
            .await
            .map_err(|e| BatchError::new(e.to_string()))?;

        let mut out = Vec::new();
        for key in keys {
            if let Some(value) = editor.get_text(key.as_str()) {
                out.push((key.clone(), value.to_string()));
            }
        }
        Ok(out)
    })
}

/// Applies `tags` to the media file at `path` using the high-level
/// [`BatchMetadataEditor`].
///
/// On WASM the function always returns an error because file I/O is unavailable.
#[cfg(target_arch = "wasm32")]
fn apply_tags_to_file(_path: &Path, _tags: &[(String, String)]) -> Result<(), BatchError> {
    Err(BatchError::new("file I/O is not supported on WASM"))
}

#[cfg(not(target_arch = "wasm32"))]
fn apply_tags_to_file(path: &Path, tags: &[(String, String)]) -> Result<(), BatchError> {
    use super::tags::TagValue;
    use crate::metadata::editor::BatchMetadataEditor;

    // Validate the path exists before attempting to open it.
    if !path.exists() {
        return Err(BatchError::new(format!(
            "file not found: {}",
            path.display()
        )));
    }

    let mut editor = BatchMetadataEditor::new();
    for (key, value) in tags {
        editor = editor.set(key.as_str(), TagValue::Text(value.clone()));
    }

    editor
        .apply_to_file(path)
        .map(|_count| ())
        .map_err(|e| BatchError::new(e.to_string()))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tmp_str(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("oximedia-container-metadata-batch-{name}"))
            .to_string_lossy()
            .into_owned()
    }

    // ── BatchResult ────────────────────────────────────────────────────

    #[test]
    fn batch_result_report_all_ok() {
        let mut r = BatchResult::default();
        r.ok.push(PathBuf::from("a.flac"));
        r.ok.push(PathBuf::from("b.flac"));
        let report = r.into_report();
        assert!(report.contains("2"), "expected count '2' in '{report}'");
        assert!(
            report.contains("successfully"),
            "expected 'successfully' in '{report}'"
        );
    }

    #[test]
    fn batch_result_report_partial_failure() {
        let mut r = BatchResult::default();
        r.ok.push(PathBuf::from("ok.flac"));
        r.failed.push((
            PathBuf::from("bad.mkv"),
            BatchError::new("format unsupported"),
        ));
        let report = r.into_report();
        assert!(report.contains("1"), "should mention 1 success");
        assert!(report.contains("bad.mkv"), "should name the failed path");
        assert!(
            report.contains("format unsupported"),
            "should include the error"
        );
    }

    #[test]
    fn batch_result_report_all_failed() {
        let mut r = BatchResult::default();
        r.failed
            .push((PathBuf::from("x.wav"), BatchError::new("io error")));
        r.failed
            .push((PathBuf::from("y.ogg"), BatchError::new("crc mismatch")));
        let report = r.into_report();
        assert!(report.contains("0"), "0 succeeded");
        assert!(report.contains("2"), "2 failed");
    }

    #[test]
    fn batch_result_all_succeeded_true() {
        let mut r = BatchResult::default();
        r.ok.push(PathBuf::from("a.flac"));
        assert!(r.all_succeeded());
    }

    #[test]
    fn batch_result_all_succeeded_false() {
        let mut r = BatchResult::default();
        r.ok.push(PathBuf::from("a.flac"));
        r.failed
            .push((PathBuf::from("b.flac"), BatchError::new("err")));
        assert!(!r.all_succeeded());
    }

    #[test]
    fn batch_result_total() {
        let mut r = BatchResult::default();
        r.ok.push(PathBuf::from("a.flac"));
        r.ok.push(PathBuf::from("b.flac"));
        r.failed
            .push((PathBuf::from("c.flac"), BatchError::new("err")));
        assert_eq!(r.total(), 3);
    }

    // ── BatchMetadataUpdate builder ────────────────────────────────────

    #[test]
    fn builder_default_is_empty() {
        let b = BatchMetadataUpdate::new();
        assert!(b.files.is_empty());
        assert!(b.tags.is_empty());
        assert!(b.copy_from.is_none());
    }

    #[test]
    fn builder_add_file() {
        let b = BatchMetadataUpdate::new()
            .add_file("a.flac")
            .add_file("b.flac");
        assert_eq!(b.files.len(), 2);
    }

    #[test]
    fn builder_set_tag() {
        let b = BatchMetadataUpdate::new()
            .set_tag("TITLE", "Hello")
            .set_tag("ARTIST", "World");
        assert_eq!(b.tags.len(), 2);
        assert_eq!(b.tags[0], ("TITLE".to_string(), "Hello".to_string()));
        assert_eq!(b.tags[1], ("ARTIST".to_string(), "World".to_string()));
    }

    #[test]
    fn builder_copy_from() {
        let src = tmp_str("source.flac");
        let b = BatchMetadataUpdate::new().copy_from(&src, vec!["TITLE".into(), "ARTIST".into()]);
        let (path, keys) = b.copy_from.as_ref().expect("copy_from should be set");
        assert_eq!(path, Path::new(&src));
        assert_eq!(keys, &["TITLE", "ARTIST"]);
    }

    // ── apply with non-existent paths (forced failure) ─────────────────

    #[test]
    fn apply_nonexistent_path_reports_failure() {
        let result = BatchMetadataUpdate::new()
            .add_file(tmp_str("definitely_does_not_exist.flac"))
            .set_tag("TITLE", "Test")
            .apply();
        assert_eq!(result.ok.len(), 0);
        assert_eq!(result.failed.len(), 1);
    }

    #[test]
    fn apply_mixed_paths_collects_failures() {
        // Two nonexistent paths → both fail; zero succeed.
        let result = BatchMetadataUpdate::new()
            .add_file(tmp_str("missing_a.flac"))
            .add_file(tmp_str("missing_b.flac"))
            .set_tag("ALBUM", "Test Album")
            .apply();
        assert_eq!(result.ok.len(), 0, "no files should succeed");
        assert_eq!(result.failed.len(), 2, "both files should fail");
        let report = result.into_report();
        assert!(report.contains("2"), "report should mention 2 failures");
    }

    #[test]
    fn apply_no_files_returns_empty_result() {
        let result = BatchMetadataUpdate::new()
            .set_tag("TITLE", "Unused")
            .apply();
        assert_eq!(result.ok.len(), 0);
        assert_eq!(result.failed.len(), 0);
        assert_eq!(result.total(), 0);
    }

    // ── build_merged_tags order ────────────────────────────────────────

    #[test]
    fn merged_tags_explicit_only() {
        let b = BatchMetadataUpdate::new().set_tag("TITLE", "Explicit");
        let merged = b
            .build_merged_tags()
            .expect("should not fail without copy_from");
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0], ("TITLE".to_string(), "Explicit".to_string()));
    }

    #[test]
    fn merged_tags_empty_when_no_config() {
        let b = BatchMetadataUpdate::new();
        let merged = b.build_merged_tags().expect("should not fail");
        assert!(merged.is_empty());
    }
}
