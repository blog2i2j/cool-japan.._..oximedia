//! Batch EDL export functionality.
//!
//! Provides types and operations for exporting multiple EDL sequences
//! in various formats as a batch operation.
//!
//! The [`BatchEdlExporter`] type offers a [`BatchEdlExporter::export_parallel`]
//! method that processes multiple [`crate::Edl`] structs concurrently via rayon.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use crate::error::{EdlError, EdlResult};
use crate::{Edl, EdlGenerator};
use rayon::prelude::*;
use std::path::{Path, PathBuf};

/// Supported export formats for EDL batch export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// CMX 3600 EDL format.
    Cmx3600,
    /// Final Cut Pro XML format.
    FcpXml,
    /// DaVinci Resolve EDL format.
    DavinciResolveEdl,
    /// Open Timeline IO format.
    Otio,
    /// Comma-separated values format.
    Csv,
}

impl ExportFormat {
    /// Returns the file extension for this format (without leading dot).
    #[must_use]
    pub fn extension(&self) -> &str {
        match self {
            Self::Cmx3600 => "edl",
            Self::FcpXml => "xml",
            Self::DavinciResolveEdl => "edl",
            Self::Otio => "otio",
            Self::Csv => "csv",
        }
    }

    /// Returns true if this format is XML-based.
    #[must_use]
    pub fn is_xml(&self) -> bool {
        matches!(self, Self::FcpXml)
    }

    /// Returns the human-readable name of the format.
    #[must_use]
    pub fn display_name(&self) -> &str {
        match self {
            Self::Cmx3600 => "CMX 3600",
            Self::FcpXml => "Final Cut Pro XML",
            Self::DavinciResolveEdl => "DaVinci Resolve EDL",
            Self::Otio => "Open Timeline IO",
            Self::Csv => "CSV",
        }
    }
}

impl std::fmt::Display for ExportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// A single item in a batch export queue.
#[derive(Debug, Clone)]
pub struct BatchExportItem {
    /// Name of the sequence to export.
    pub sequence_name: String,
    /// Format to export to.
    pub format: ExportFormat,
    /// Output file path (directory or full path).
    pub output_path: String,
    /// Frame rate for timecode calculations.
    pub frame_rate: f32,
}

impl BatchExportItem {
    /// Create a new batch export item.
    #[must_use]
    pub fn new(
        sequence_name: impl Into<String>,
        format: ExportFormat,
        output_path: impl Into<String>,
        frame_rate: f32,
    ) -> Self {
        Self {
            sequence_name: sequence_name.into(),
            format,
            output_path: output_path.into(),
            frame_rate,
        }
    }

    /// Generate the output filename for this export item.
    ///
    /// The filename is `{sequence_name}.{extension}`.
    #[must_use]
    pub fn filename(&self) -> String {
        format!("{}.{}", self.sequence_name, self.format.extension())
    }

    /// Returns the full output file path combining `output_path` and `filename()`.
    #[must_use]
    pub fn full_output_path(&self) -> String {
        format!("{}/{}", self.output_path, self.filename())
    }
}

/// A queue of batch export items to be processed.
#[derive(Debug, Default)]
pub struct BatchExportQueue {
    /// Items in the queue.
    pub items: Vec<BatchExportItem>,
}

impl BatchExportQueue {
    /// Create a new empty batch export queue.
    #[must_use]
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// Add an item to the queue.
    pub fn add(&mut self, item: BatchExportItem) {
        self.items.push(item);
    }

    /// Remove all items for a given sequence name.
    ///
    /// Returns the number of items removed.
    pub fn remove_by_sequence(&mut self, sequence_name: &str) -> usize {
        let before = self.items.len();
        self.items.retain(|i| i.sequence_name != sequence_name);
        before - self.items.len()
    }

    /// Returns the number of items in the queue.
    #[must_use]
    pub fn item_count(&self) -> usize {
        self.items.len()
    }

    /// Returns all items that match the given export format.
    #[must_use]
    pub fn items_for_format(&self, fmt: &ExportFormat) -> Vec<&BatchExportItem> {
        self.items.iter().filter(|i| &i.format == fmt).collect()
    }

    /// Returns true if the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Clear all items from the queue.
    pub fn clear(&mut self) {
        self.items.clear();
    }

    /// Returns all unique sequence names in the queue.
    #[must_use]
    pub fn sequence_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self
            .items
            .iter()
            .map(|i| i.sequence_name.as_str())
            .collect();
        names.dedup();
        names
    }
}

/// Result of a single export operation.
#[derive(Debug, Clone)]
pub struct ExportResult {
    /// Name of the sequence that was exported.
    pub sequence_name: String,
    /// Whether the export succeeded.
    pub success: bool,
    /// Number of bytes written (0 on failure).
    pub bytes_written: u64,
    /// Optional error message on failure.
    pub error_msg: Option<String>,
}

impl ExportResult {
    /// Create a successful export result.
    #[must_use]
    pub fn success(sequence_name: impl Into<String>, bytes_written: u64) -> Self {
        Self {
            sequence_name: sequence_name.into(),
            success: true,
            bytes_written,
            error_msg: None,
        }
    }

    /// Create a failed export result.
    #[must_use]
    pub fn failure(sequence_name: impl Into<String>, error_msg: impl Into<String>) -> Self {
        Self {
            sequence_name: sequence_name.into(),
            success: false,
            bytes_written: 0,
            error_msg: Some(error_msg.into()),
        }
    }

    /// Returns true if the export was successful.
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.success
    }

    /// Returns the error message if the export failed.
    #[must_use]
    pub fn error(&self) -> Option<&str> {
        self.error_msg.as_deref()
    }
}

/// Summary of a batch export run.
#[derive(Debug, Default)]
pub struct BatchExportSummary {
    /// Individual results for each export item.
    pub results: Vec<ExportResult>,
}

impl BatchExportSummary {
    /// Create a new empty batch export summary.
    #[must_use]
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// Add a result to the summary.
    pub fn add_result(&mut self, result: ExportResult) {
        self.results.push(result);
    }

    /// Returns the number of successful exports.
    #[must_use]
    pub fn success_count(&self) -> usize {
        self.results.iter().filter(|r| r.success).count()
    }

    /// Returns the number of failed exports.
    #[must_use]
    pub fn failure_count(&self) -> usize {
        self.results.iter().filter(|r| !r.success).count()
    }

    /// Returns the total bytes written across all successful exports.
    #[must_use]
    pub fn total_bytes_written(&self) -> u64 {
        self.results.iter().map(|r| r.bytes_written).sum()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// BatchEdlExporter — parallel multi-EDL export
// ────────────────────────────────────────────────────────────────────────────

/// Exports multiple [`Edl`] documents to individual files in an output
/// directory, optionally in parallel via rayon.
///
/// Each EDL is written to `<output_dir>/<title_or_index>.edl`.  When the EDL
/// has no title, the file is named `edl_<index>.edl`.
#[derive(Debug, Default)]
pub struct BatchEdlExporter {
    /// EDL generator used for serialisation.
    generator: EdlGenerator,
}

impl BatchEdlExporter {
    /// Create a new exporter with default generator settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            generator: EdlGenerator::new(),
        }
    }

    /// Create a new exporter with a custom generator.
    #[must_use]
    pub fn with_generator(generator: EdlGenerator) -> Self {
        Self { generator }
    }

    /// Export a collection of EDLs to `output_dir` using rayon parallel iteration.
    ///
    /// Each element of the returned `Vec` corresponds positionally to an EDL in
    /// `edls`.  A successful export contains the [`PathBuf`] of the written
    /// file; a failed export contains the [`EdlError`] that caused the failure.
    ///
    /// The method creates `output_dir` and any necessary parent directories
    /// before writing.  Individual file errors do **not** abort the batch —
    /// every EDL is attempted regardless of other failures.
    pub fn export_parallel(&self, edls: Vec<Edl>, output_dir: &Path) -> Vec<EdlResult<PathBuf>> {
        // Create output directory upfront; propagate as errors for every item if
        // this fails (we cannot write any file without the directory).
        if let Err(io_err) = std::fs::create_dir_all(output_dir) {
            let wrapped = EdlError::Io(io_err);
            return edls
                .into_iter()
                .map(|_| Err(EdlError::Io(std::io::Error::other(wrapped.to_string()))))
                .collect();
        }

        let generator = &self.generator;
        let output_dir_ref = output_dir;

        // Enumerate before par_iter so each item can derive its own filename.
        let indexed: Vec<(usize, Edl)> = edls.into_iter().enumerate().collect();

        indexed
            .into_par_iter()
            .map(|(idx, edl)| {
                let stem = edl
                    .title
                    .as_deref()
                    .map(sanitize_filename)
                    .unwrap_or_else(|| format!("edl_{idx:04}"));

                let file_path = output_dir_ref.join(format!("{stem}.edl"));

                let content = generator.generate(&edl)?;
                std::fs::write(&file_path, &content).map_err(EdlError::Io)?;
                Ok(file_path)
            })
            .collect()
    }
}

/// Replace characters that are invalid in file names with underscores.
fn sanitize_filename(title: &str) -> String {
    title
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_format_extension() {
        assert_eq!(ExportFormat::Cmx3600.extension(), "edl");
        assert_eq!(ExportFormat::FcpXml.extension(), "xml");
        assert_eq!(ExportFormat::DavinciResolveEdl.extension(), "edl");
        assert_eq!(ExportFormat::Otio.extension(), "otio");
        assert_eq!(ExportFormat::Csv.extension(), "csv");
    }

    #[test]
    fn test_export_format_is_xml() {
        assert!(ExportFormat::FcpXml.is_xml());
        assert!(!ExportFormat::Cmx3600.is_xml());
        assert!(!ExportFormat::DavinciResolveEdl.is_xml());
        assert!(!ExportFormat::Otio.is_xml());
        assert!(!ExportFormat::Csv.is_xml());
    }

    #[test]
    fn test_export_format_display() {
        assert_eq!(ExportFormat::Cmx3600.to_string(), "CMX 3600");
        assert_eq!(ExportFormat::FcpXml.to_string(), "Final Cut Pro XML");
        assert_eq!(ExportFormat::Csv.to_string(), "CSV");
    }

    #[test]
    fn test_batch_export_item_filename() {
        let item = BatchExportItem::new("MySequence", ExportFormat::Cmx3600, "/output", 25.0);
        assert_eq!(item.filename(), "MySequence.edl");
    }

    #[test]
    fn test_batch_export_item_filename_xml() {
        let item = BatchExportItem::new("Project_01", ExportFormat::FcpXml, "/exports", 29.97);
        assert_eq!(item.filename(), "Project_01.xml");
    }

    #[test]
    fn test_batch_export_item_full_output_path() {
        let out_dir = std::env::temp_dir()
            .join("oximedia-edl-batch-exports")
            .to_string_lossy()
            .into_owned();
        let item = BatchExportItem::new("Seq1", ExportFormat::Csv, &out_dir, 24.0);
        assert_eq!(item.full_output_path(), format!("{out_dir}/Seq1.csv"));
    }

    #[test]
    fn test_batch_export_queue_add_and_count() {
        let mut queue = BatchExportQueue::new();
        assert_eq!(queue.item_count(), 0);
        assert!(queue.is_empty());

        queue.add(BatchExportItem::new(
            "Seq1",
            ExportFormat::Cmx3600,
            "/out",
            25.0,
        ));
        queue.add(BatchExportItem::new(
            "Seq2",
            ExportFormat::FcpXml,
            "/out",
            25.0,
        ));

        assert_eq!(queue.item_count(), 2);
        assert!(!queue.is_empty());
    }

    #[test]
    fn test_batch_export_queue_remove_by_sequence() {
        let mut queue = BatchExportQueue::new();
        queue.add(BatchExportItem::new(
            "Seq1",
            ExportFormat::Cmx3600,
            "/out",
            25.0,
        ));
        queue.add(BatchExportItem::new(
            "Seq1",
            ExportFormat::FcpXml,
            "/out",
            25.0,
        ));
        queue.add(BatchExportItem::new(
            "Seq2",
            ExportFormat::Csv,
            "/out",
            25.0,
        ));

        let removed = queue.remove_by_sequence("Seq1");
        assert_eq!(removed, 2);
        assert_eq!(queue.item_count(), 1);
    }

    #[test]
    fn test_batch_export_queue_items_for_format() {
        let mut queue = BatchExportQueue::new();
        queue.add(BatchExportItem::new(
            "Seq1",
            ExportFormat::Cmx3600,
            "/out",
            25.0,
        ));
        queue.add(BatchExportItem::new(
            "Seq2",
            ExportFormat::Cmx3600,
            "/out",
            25.0,
        ));
        queue.add(BatchExportItem::new(
            "Seq3",
            ExportFormat::FcpXml,
            "/out",
            25.0,
        ));

        let cmx_items = queue.items_for_format(&ExportFormat::Cmx3600);
        assert_eq!(cmx_items.len(), 2);

        let xml_items = queue.items_for_format(&ExportFormat::FcpXml);
        assert_eq!(xml_items.len(), 1);
        assert_eq!(xml_items[0].sequence_name, "Seq3");
    }

    #[test]
    fn test_batch_export_queue_clear() {
        let mut queue = BatchExportQueue::new();
        queue.add(BatchExportItem::new(
            "Seq1",
            ExportFormat::Cmx3600,
            "/out",
            25.0,
        ));
        queue.add(BatchExportItem::new(
            "Seq2",
            ExportFormat::FcpXml,
            "/out",
            25.0,
        ));
        queue.clear();
        assert!(queue.is_empty());
    }

    #[test]
    fn test_export_result_success() {
        let result = ExportResult::success("MySeq", 4096);
        assert!(result.is_success());
        assert_eq!(result.bytes_written, 4096);
        assert_eq!(result.sequence_name, "MySeq");
        assert!(result.error().is_none());
    }

    #[test]
    fn test_export_result_failure() {
        let result = ExportResult::failure("BadSeq", "File not found");
        assert!(!result.is_success());
        assert_eq!(result.bytes_written, 0);
        assert_eq!(result.error(), Some("File not found"));
    }

    #[test]
    fn test_batch_export_summary() {
        let mut summary = BatchExportSummary::new();
        summary.add_result(ExportResult::success("S1", 1024));
        summary.add_result(ExportResult::success("S2", 2048));
        summary.add_result(ExportResult::failure("S3", "error"));

        assert_eq!(summary.success_count(), 2);
        assert_eq!(summary.failure_count(), 1);
        assert_eq!(summary.total_bytes_written(), 3072);
    }

    #[test]
    fn test_export_format_equality() {
        assert_eq!(ExportFormat::Cmx3600, ExportFormat::Cmx3600);
        assert_ne!(ExportFormat::Cmx3600, ExportFormat::FcpXml);
    }

    // ── BatchEdlExporter tests ───────────────────────────────────────────────

    use crate::event::{EditType, EdlEvent, TrackType};
    use crate::timecode::{EdlFrameRate, EdlTimecode};
    use crate::EdlFormat;

    fn make_test_edl(title: &str, reel: &str) -> Edl {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_title(title.to_string());
        edl.set_frame_rate(EdlFrameRate::Fps25);
        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("valid tc");
        let tc2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("valid tc");
        let event = EdlEvent::new(
            1,
            reel.to_string(),
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );
        edl.add_event(event).expect("add event");
        edl
    }

    #[test]
    fn test_batch_edl_exporter_export_parallel() {
        let output_dir = std::env::temp_dir().join("oximedia_edl_batch_test");

        let edls = vec![
            make_test_edl("Alpha Sequence", "A001"),
            make_test_edl("Beta Sequence", "B001"),
            make_test_edl("Gamma Sequence", "C001"),
        ];

        let exporter = BatchEdlExporter::new();
        let results = exporter.export_parallel(edls, &output_dir);

        assert_eq!(results.len(), 3, "should have one result per EDL");

        for result in &results {
            assert!(result.is_ok(), "export should succeed: {result:?}");
        }

        let paths: Vec<PathBuf> = results.into_iter().map(|r| r.expect("ok")).collect();

        // All files should exist and be non-empty.
        for path in &paths {
            assert!(path.exists(), "output file should exist: {path:?}");
            let content = std::fs::read_to_string(path).expect("read file");
            assert!(!content.is_empty(), "exported EDL should not be empty");
        }

        // Clean up.
        for path in &paths {
            let _ = std::fs::remove_file(path);
        }
        let _ = std::fs::remove_dir(&output_dir);
    }

    #[test]
    fn test_batch_edl_exporter_unnamed_edls() {
        let output_dir = std::env::temp_dir().join("oximedia_edl_batch_unnamed_test");

        // EDLs without titles should receive auto-generated filenames.
        let edls = vec![Edl::new(EdlFormat::Cmx3600), Edl::new(EdlFormat::Cmx3600)];

        let exporter = BatchEdlExporter::new();
        let results = exporter.export_parallel(edls, &output_dir);

        assert_eq!(results.len(), 2);
        for result in &results {
            assert!(
                result.is_ok(),
                "export of untitled EDL should succeed: {result:?}"
            );
        }

        let paths: Vec<PathBuf> = results.into_iter().map(|r| r.expect("ok")).collect();
        for path in &paths {
            assert!(path.exists());
            let _ = std::fs::remove_file(path);
        }
        let _ = std::fs::remove_dir(&output_dir);
    }
}
