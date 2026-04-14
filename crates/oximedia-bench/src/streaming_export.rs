//! Streaming CSV and JSON export for benchmark results.
//!
//! Unlike the batch export methods on [`BenchmarkResults`](crate::BenchmarkResults),
//! this module writes results incrementally to an [`io::Write`] sink so that
//! memory usage remains constant regardless of the number of codec/sequence
//! pairs.  This is particularly useful for very large benchmark suites with
//! thousands of sequences.
//!
//! # Formats
//!
//! - **CSV** — one row per `(codec, sequence)` pair, header written once.
//! - **JSON Lines (NDJSON)** — one JSON object per row, no framing array.
//!
//! # Example
//!
//! ```
//! use oximedia_bench::streaming_export::{StreamingCsvWriter, StreamingJsonWriter};
//! use oximedia_bench::streaming_export::ExportRow;
//! use oximedia_core::types::CodecId;
//!
//! let mut buf = Vec::new();
//! let mut csv = StreamingCsvWriter::new(&mut buf);
//! csv.write_header().unwrap();
//!
//! let row = ExportRow {
//!     codec: "Av1".to_string(),
//!     preset: Some("medium".to_string()),
//!     bitrate_kbps: Some(2000),
//!     sequence_name: "clip_720p".to_string(),
//!     encoding_fps: 45.0,
//!     decoding_fps: 120.0,
//!     file_size_bytes: 1_500_000,
//!     psnr_db: Some(38.5),
//!     ssim: Some(0.965),
//!     vmaf: None,
//! };
//! csv.write_row(&row).unwrap();
//! csv.flush().unwrap();
//! let output = String::from_utf8(buf).unwrap();
//! assert!(output.contains("clip_720p"));
//! ```

use crate::{BenchError, BenchResult, BenchmarkResults};
use serde::{Deserialize, Serialize};
use std::io::{self, Write};

// ---------------------------------------------------------------------------
// Flat row representation
// ---------------------------------------------------------------------------

/// A single flattened export row representing one `(codec, sequence)` pair.
///
/// This is the common denominator for both CSV and JSON-lines streaming export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportRow {
    /// Codec identifier string (e.g. `"Av1"`).
    pub codec: String,
    /// Preset label, if any.
    pub preset: Option<String>,
    /// Target bitrate in kbps, if any.
    pub bitrate_kbps: Option<u32>,
    /// Test sequence name.
    pub sequence_name: String,
    /// Encoding throughput in frames per second.
    pub encoding_fps: f64,
    /// Decoding throughput in frames per second.
    pub decoding_fps: f64,
    /// Encoded file size in bytes.
    pub file_size_bytes: u64,
    /// PSNR in dB, if computed.
    pub psnr_db: Option<f64>,
    /// SSIM (0..1), if computed.
    pub ssim: Option<f64>,
    /// VMAF (0..100), if computed.
    pub vmaf: Option<f64>,
}

impl ExportRow {
    /// Create rows from a [`BenchmarkResults`] instance without collecting them
    /// all into a single `Vec` first.
    ///
    /// This is an iterator adapter that yields one [`ExportRow`] per sequence
    /// result.
    pub fn iter_from_results(results: &BenchmarkResults) -> impl Iterator<Item = ExportRow> + '_ {
        results.codec_results.iter().flat_map(|codec| {
            codec.sequence_results.iter().map(move |seq| ExportRow {
                codec: format!("{:?}", codec.codec_id),
                preset: codec.preset.clone(),
                bitrate_kbps: codec.bitrate_kbps,
                sequence_name: seq.sequence_name.clone(),
                encoding_fps: seq.encoding_fps,
                decoding_fps: seq.decoding_fps,
                file_size_bytes: seq.file_size_bytes,
                psnr_db: seq.metrics.psnr,
                ssim: seq.metrics.ssim,
                vmaf: seq.metrics.vmaf,
            })
        })
    }
}

// ---------------------------------------------------------------------------
// Streaming CSV writer
// ---------------------------------------------------------------------------

/// Writes benchmark rows incrementally in CSV format.
pub struct StreamingCsvWriter<W: Write> {
    writer: W,
    header_written: bool,
}

impl<W: Write> StreamingCsvWriter<W> {
    /// Wrap a writer.
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            header_written: false,
        }
    }

    /// Write the CSV header row.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if writing fails.
    pub fn write_header(&mut self) -> BenchResult<()> {
        writeln!(
            self.writer,
            "Codec,Preset,Bitrate (kbps),Sequence,Encoding FPS,Decoding FPS,\
             File Size (bytes),PSNR (dB),SSIM,VMAF"
        )
        .map_err(BenchError::Io)?;
        self.header_written = true;
        Ok(())
    }

    /// Write a single data row.
    ///
    /// If the header has not been written yet it is written automatically.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if writing fails.
    pub fn write_row(&mut self, row: &ExportRow) -> BenchResult<()> {
        if !self.header_written {
            self.write_header()?;
        }
        writeln!(
            self.writer,
            "{},{},{},{},{:.2},{:.2},{},{},{},{}",
            csv_escape(&row.codec),
            csv_escape(row.preset.as_deref().unwrap_or("")),
            row.bitrate_kbps.map_or(String::new(), |b| b.to_string()),
            csv_escape(&row.sequence_name),
            row.encoding_fps,
            row.decoding_fps,
            row.file_size_bytes,
            opt_f64(row.psnr_db),
            opt_f64(row.ssim),
            opt_f64(row.vmaf),
        )
        .map_err(BenchError::Io)?;
        Ok(())
    }

    /// Write all rows produced by [`ExportRow::iter_from_results`].
    ///
    /// # Errors
    ///
    /// Returns an error if any write operation fails.
    pub fn write_all(&mut self, results: &BenchmarkResults) -> BenchResult<()> {
        for row in ExportRow::iter_from_results(results) {
            self.write_row(&row)?;
        }
        Ok(())
    }

    /// Flush the underlying writer.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if flushing fails.
    pub fn flush(&mut self) -> BenchResult<()> {
        self.writer.flush().map_err(BenchError::Io)
    }

    /// Return whether the header has been written.
    #[must_use]
    pub fn header_written(&self) -> bool {
        self.header_written
    }
}

// ---------------------------------------------------------------------------
// Streaming JSON-lines writer
// ---------------------------------------------------------------------------

/// Writes benchmark rows incrementally in JSON Lines (NDJSON) format.
///
/// Each row is written as a self-contained JSON object on its own line,
/// so the output can be processed one line at a time by downstream tools.
pub struct StreamingJsonWriter<W: Write> {
    writer: W,
    rows_written: u64,
}

impl<W: Write> StreamingJsonWriter<W> {
    /// Wrap a writer.
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            rows_written: 0,
        }
    }

    /// Write a single row as a JSON object followed by a newline.
    ///
    /// # Errors
    ///
    /// Returns an error if serialisation or writing fails.
    pub fn write_row(&mut self, row: &ExportRow) -> BenchResult<()> {
        let json = serde_json::to_string(row)?;
        writeln!(self.writer, "{json}").map_err(BenchError::Io)?;
        self.rows_written += 1;
        Ok(())
    }

    /// Write all rows produced by [`ExportRow::iter_from_results`].
    ///
    /// # Errors
    ///
    /// Returns an error if any write or serialisation fails.
    pub fn write_all(&mut self, results: &BenchmarkResults) -> BenchResult<()> {
        for row in ExportRow::iter_from_results(results) {
            self.write_row(&row)?;
        }
        Ok(())
    }

    /// Flush the underlying writer.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if flushing fails.
    pub fn flush(&mut self) -> BenchResult<()> {
        self.writer.flush().map_err(BenchError::Io)
    }

    /// Number of rows written so far.
    #[must_use]
    pub fn rows_written(&self) -> u64 {
        self.rows_written
    }
}

// ---------------------------------------------------------------------------
// Convenience: write to file paths
// ---------------------------------------------------------------------------

/// Stream benchmark results to a CSV file.
///
/// # Errors
///
/// Returns an error if the file cannot be created or written.
pub fn stream_csv_to_file(
    results: &BenchmarkResults,
    path: impl AsRef<std::path::Path>,
) -> BenchResult<()> {
    let file = std::fs::File::create(path)?;
    let buf = io::BufWriter::new(file);
    let mut writer = StreamingCsvWriter::new(buf);
    writer.write_header()?;
    writer.write_all(results)?;
    writer.flush()?;
    Ok(())
}

/// Stream benchmark results to a JSON Lines file.
///
/// # Errors
///
/// Returns an error if the file cannot be created or written.
pub fn stream_json_to_file(
    results: &BenchmarkResults,
    path: impl AsRef<std::path::Path>,
) -> BenchResult<()> {
    let file = std::fs::File::create(path)?;
    let buf = io::BufWriter::new(file);
    let mut writer = StreamingJsonWriter::new(buf);
    writer.write_all(results)?;
    writer.flush()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Minimal CSV field escaping: if the value contains a comma, quote, or newline
/// it is wrapped in double quotes with internal quotes doubled.
fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        let escaped = s.replace('"', "\"\"");
        format!("\"{escaped}\"")
    } else {
        s.to_string()
    }
}

fn opt_f64(v: Option<f64>) -> String {
    v.map_or(String::new(), |val| format!("{val:.4}"))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::QualityMetrics;
    use crate::stats::Statistics;
    use crate::{BenchmarkConfig, CodecBenchmarkResult, SequenceResult};
    use oximedia_core::types::CodecId;
    use std::time::Duration;

    fn sample_results() -> BenchmarkResults {
        let seq1 = SequenceResult {
            sequence_name: "scene_a".to_string(),
            frames_processed: 100,
            encoding_fps: 30.0,
            decoding_fps: 200.0,
            file_size_bytes: 500_000,
            metrics: QualityMetrics {
                psnr: Some(36.0),
                ssim: Some(0.95),
                vmaf: None,
                mse: None,
                psnr_y: None,
                psnr_u: None,
                psnr_v: None,
                ssim_y: None,
                ssim_u: None,
                ssim_v: None,
            },
            encoding_duration: Duration::from_secs(3),
            decoding_duration: Duration::from_secs(1),
        };
        let seq2 = SequenceResult {
            sequence_name: "scene_b".to_string(),
            frames_processed: 200,
            encoding_fps: 50.0,
            decoding_fps: 300.0,
            file_size_bytes: 800_000,
            metrics: QualityMetrics {
                psnr: Some(40.0),
                ssim: Some(0.98),
                vmaf: Some(92.0),
                mse: None,
                psnr_y: None,
                psnr_u: None,
                psnr_v: None,
                ssim_y: None,
                ssim_u: None,
                ssim_v: None,
            },
            encoding_duration: Duration::from_secs(4),
            decoding_duration: Duration::from_secs(1),
        };
        let codec = CodecBenchmarkResult {
            codec_id: CodecId::Av1,
            preset: Some("medium".to_string()),
            bitrate_kbps: Some(2000),
            cq_level: None,
            sequence_results: vec![seq1, seq2],
            statistics: Statistics::default(),
        };
        BenchmarkResults {
            codec_results: vec![codec],
            timestamp: "2025-06-01".to_string(),
            total_duration: Duration::from_secs(10),
            config: BenchmarkConfig::default(),
        }
    }

    #[test]
    fn test_csv_streaming_header() {
        let mut buf = Vec::new();
        let mut w = StreamingCsvWriter::new(&mut buf);
        assert!(!w.header_written());
        w.write_header().expect("write header failed in test");
        assert!(w.header_written());
        let output = String::from_utf8(buf).expect("utf8 failed in test");
        assert!(output.starts_with("Codec,"));
    }

    #[test]
    fn test_csv_streaming_row() {
        let mut buf = Vec::new();
        {
            let mut w = StreamingCsvWriter::new(&mut buf);
            let row = ExportRow {
                codec: "Av1".to_string(),
                preset: Some("fast".to_string()),
                bitrate_kbps: Some(1000),
                sequence_name: "seq1".to_string(),
                encoding_fps: 60.0,
                decoding_fps: 240.0,
                file_size_bytes: 123_456,
                psnr_db: Some(35.5),
                ssim: Some(0.94),
                vmaf: None,
            };
            w.write_row(&row).expect("write row failed in test");
            w.flush().expect("flush failed in test");
        }
        let output = String::from_utf8(buf).expect("utf8 failed in test");
        // Should auto-write header
        assert!(output.contains("Codec,"));
        assert!(output.contains("seq1"));
        assert!(output.contains("60.00"));
    }

    #[test]
    fn test_csv_write_all() {
        let results = sample_results();
        let mut buf = Vec::new();
        {
            let mut w = StreamingCsvWriter::new(&mut buf);
            w.write_header().expect("write header failed in test");
            w.write_all(&results).expect("write all failed in test");
            w.flush().expect("flush failed in test");
        }
        let output = String::from_utf8(buf).expect("utf8 failed in test");
        assert!(output.contains("scene_a"));
        assert!(output.contains("scene_b"));
        // Count lines: header + 2 data rows
        let line_count = output.lines().count();
        assert_eq!(line_count, 3);
    }

    #[test]
    fn test_json_streaming_row() {
        let mut buf = Vec::new();
        {
            let mut w = StreamingJsonWriter::new(&mut buf);
            let row = ExportRow {
                codec: "Vp9".to_string(),
                preset: None,
                bitrate_kbps: None,
                sequence_name: "clip_4k".to_string(),
                encoding_fps: 10.0,
                decoding_fps: 80.0,
                file_size_bytes: 5_000_000,
                psnr_db: None,
                ssim: None,
                vmaf: None,
            };
            w.write_row(&row).expect("write row failed in test");
            assert_eq!(w.rows_written(), 1);
            w.flush().expect("flush failed in test");
        }
        let output = String::from_utf8(buf).expect("utf8 failed in test");
        // Should be valid JSON
        let parsed: serde_json::Value =
            serde_json::from_str(output.trim()).expect("parse json failed in test");
        assert_eq!(parsed["codec"], "Vp9");
        assert_eq!(parsed["sequence_name"], "clip_4k");
    }

    #[test]
    fn test_json_write_all() {
        let results = sample_results();
        let mut buf = Vec::new();
        {
            let mut w = StreamingJsonWriter::new(&mut buf);
            w.write_all(&results).expect("write all failed in test");
            assert_eq!(w.rows_written(), 2);
            w.flush().expect("flush failed in test");
        }
        let output = String::from_utf8(buf).expect("utf8 failed in test");
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines.len(), 2);
        for line in lines {
            let parsed: serde_json::Value =
                serde_json::from_str(line).expect("parse json line failed in test");
            assert!(parsed["encoding_fps"].is_number());
        }
    }

    #[test]
    fn test_stream_csv_to_file() {
        let results = sample_results();
        let path = std::env::temp_dir().join("oximedia_bench_stream_csv_test.csv");
        stream_csv_to_file(&results, &path).expect("stream csv to file failed in test");
        let content = std::fs::read_to_string(&path).expect("read back failed in test");
        assert!(content.contains("scene_a"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_stream_json_to_file() {
        let results = sample_results();
        let path = std::env::temp_dir().join("oximedia_bench_stream_json_test.ndjson");
        stream_json_to_file(&results, &path).expect("stream json to file failed in test");
        let content = std::fs::read_to_string(&path).expect("read back failed in test");
        assert!(content.contains("scene_b"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_csv_escape_plain() {
        assert_eq!(csv_escape("hello"), "hello");
    }

    #[test]
    fn test_csv_escape_comma() {
        assert_eq!(csv_escape("a,b"), "\"a,b\"");
    }

    #[test]
    fn test_csv_escape_quotes() {
        assert_eq!(csv_escape("say \"hi\""), "\"say \"\"hi\"\"\"");
    }

    #[test]
    fn test_export_row_iter_empty() {
        let results = BenchmarkResults {
            codec_results: vec![],
            timestamp: String::new(),
            total_duration: Duration::ZERO,
            config: BenchmarkConfig::default(),
        };
        let rows: Vec<_> = ExportRow::iter_from_results(&results).collect();
        assert!(rows.is_empty());
    }
}
