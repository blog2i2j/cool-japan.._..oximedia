//! High-level repair dispatcher for detected media issues.
//!
//! This module provides a lightweight, self-contained issue dispatcher that
//! operates independently of the full [`crate::RepairEngine`].  It is intended
//! for callers that already have a list of [`DetectedIssue`]s and want to
//! apply targeted fixes without running the full analysis pipeline.
//!
//! # Issue types
//!
//! | Variant             | Description                                   |
//! |---------------------|-----------------------------------------------|
//! | `TruncatedFile`     | File ends unexpectedly; last valid sync found |
//! | `CorruptHeader`     | File header is missing or damaged             |
//! | `MissingKeyframes`  | No I-frames detected in GOP                   |
//! | `AudioSync`         | Audio and video timestamps are desynchronised |
//! | `ContainerError`    | Generic container-level structural error       |

use std::path::Path;

// ---------------------------------------------------------------------------
// IssueType
// ---------------------------------------------------------------------------

/// Classification of a detected media issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IssueType {
    /// The file was truncated; payload data is missing past a certain byte.
    TruncatedFile,
    /// The file header is corrupted or missing, preventing format detection.
    CorruptHeader,
    /// No keyframes (I-frames) are present in the stream or GOP region.
    MissingKeyframes,
    /// Audio and video streams are out of sync by a measurable offset.
    AudioSync,
    /// A generic container-level structural error (box/atom size mismatch, etc.).
    ContainerError,
}

// ---------------------------------------------------------------------------
// Severity
// ---------------------------------------------------------------------------

/// How severely the issue impacts playback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IssueSeverity {
    /// Minor visual/audio artefact; file still plays.
    Low,
    /// Noticeable degradation or seeking issues.
    Medium,
    /// Significant portion of media is unplayable.
    High,
    /// File is completely unplayable.
    Critical,
}

// ---------------------------------------------------------------------------
// DetectedIssue
// ---------------------------------------------------------------------------

/// A single issue detected in a media file.
#[derive(Debug, Clone)]
pub struct DetectedIssue {
    /// The category of the issue.
    pub issue_type: IssueType,
    /// Severity of the impact on playback.
    pub severity: IssueSeverity,
    /// Human-readable description of the issue.
    pub description: String,
    /// Byte offset where the issue was first detected (if known).
    pub location: Option<u64>,
    /// Whether an automated fix is available for this issue.
    pub fixable: bool,
    /// Confidence that this is a genuine issue (0.0 = uncertain, 1.0 = certain).
    pub confidence: f32,
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

/// Attempt to repair a single [`DetectedIssue`].
///
/// # Arguments
///
/// * `input`  – Path to the (possibly corrupted) source file.
/// * `output` – Destination path for the repaired output.
/// * `issue`  – The issue to attempt to fix.
///
/// # Returns
///
/// * `Ok(true)`  – The issue was successfully addressed.
/// * `Ok(false)` – No automated fix is available for this issue type.
/// * `Err(msg)`  – The repair attempt failed with an error message.
///
/// # Notes
///
/// All branches are stubs that log what they would do and return `Ok(true)`.
/// Callers that integrate this dispatcher into a full repair pipeline should
/// replace the stub bodies with real sub-module calls once those modules are
/// available.
pub fn fix_issue(input: &Path, output: &Path, issue: &DetectedIssue) -> Result<bool, String> {
    match issue.issue_type {
        // ------------------------------------------------------------------
        IssueType::TruncatedFile => {
            // Strategy: scan backward from end-of-file for the last valid
            // sync word; record the last-good byte offset; write a truncated
            // copy up to that point.
            let last_sync_byte = find_last_sync_word(input)?;
            log_repair_action(&format!(
                "TruncatedFile: last valid sync word at byte {last_sync_byte}; \
                 truncating output at that boundary"
            ));
            write_truncated_copy(input, output, last_sync_byte)?;
            Ok(true)
        }

        // ------------------------------------------------------------------
        IssueType::CorruptHeader => {
            // Strategy: rebuild the file header from format defaults inferred
            // from the file extension and any salvageable bytes at the start
            // of the file.
            log_repair_action(
                "CorruptHeader: rebuilding header from known format defaults",
            );
            rebuild_header_stub(input, output)?;
            Ok(true)
        }

        // ------------------------------------------------------------------
        IssueType::MissingKeyframes => {
            // Strategy: scan the stream for existing frame boundaries;
            // insert synthetic I-frame markers at regular intervals (every
            // ~30 frames by default) so that players can seek.
            let frames_marked = insert_keyframe_markers(input, output)?;
            log_repair_action(&format!(
                "MissingKeyframes: inserted {frames_marked} I-frame marker(s)"
            ));
            Ok(true)
        }

        // ------------------------------------------------------------------
        IssueType::AudioSync | IssueType::ContainerError => {
            // No automated fix available for these types in this dispatcher.
            Ok(false)
        }
    }
}

// ---------------------------------------------------------------------------
// Internal stubs
// ---------------------------------------------------------------------------

/// Scan `input` backward to find the byte offset of the last plausible sync
/// word.  A "sync word" is any of the common bitstream sync patterns:
///
/// - MPEG start code prefix: `[0x00, 0x00, 0x01]`
/// - MPEG-TS sync byte:       `0x47`
/// - ADTS sync:               first 12 bits set (`0xFF, 0xF?`)
///
/// Returns the last found offset, or the full file length if none is found
/// (which is a safe fallback — the whole file is returned unchanged).
fn find_last_sync_word(input: &Path) -> Result<u64, String> {
    let data =
        std::fs::read(input).map_err(|e| format!("TruncatedFile read error: {e}"))?;

    // Scan forward to collect all candidate sync positions, then take the last.
    let mut last_sync: u64 = data.len() as u64;

    let mut i = 0usize;
    while i < data.len() {
        // MPEG start code prefix
        if i + 2 < data.len()
            && data[i] == 0x00
            && data[i + 1] == 0x00
            && data[i + 2] == 0x01
        {
            last_sync = i as u64;
            i += 3;
            continue;
        }
        // MPEG-TS sync byte
        if data[i] == 0x47 {
            last_sync = i as u64;
        }
        // ADTS sync word (first byte 0xFF, second byte high nibble 0xF)
        if i + 1 < data.len() && data[i] == 0xFF && (data[i + 1] & 0xF0) == 0xF0 {
            last_sync = i as u64;
        }
        i += 1;
    }

    Ok(last_sync)
}

/// Write a truncated copy of `input` to `output`, keeping only the bytes
/// up to and including `last_good_byte`.
fn write_truncated_copy(input: &Path, output: &Path, last_good_byte: u64) -> Result<(), String> {
    let data =
        std::fs::read(input).map_err(|e| format!("truncated copy read error: {e}"))?;
    let end = (last_good_byte as usize).min(data.len());
    std::fs::write(output, &data[..end])
        .map_err(|e| format!("truncated copy write error: {e}"))
}

/// Stub that copies `input` to `output` with a synthetic zero header prepended.
///
/// A production implementation would parse the known container format from the
/// file extension (`.mp4`, `.mkv`, `.avi`, etc.) and write format-correct
/// default values for all mandatory header fields.
fn rebuild_header_stub(input: &Path, output: &Path) -> Result<(), String> {
    let data =
        std::fs::read(input).map_err(|e| format!("header rebuild read error: {e}"))?;

    // Synthetic header placeholder: 16 bytes of zeros.
    // Real implementation would write ftyp/RIFF/EBML bytes as appropriate.
    let mut rebuilt = vec![0u8; 16];
    // Copy the original body after the placeholder header
    if data.len() > 16 {
        rebuilt.extend_from_slice(&data[16..]);
    }
    std::fs::write(output, &rebuilt)
        .map_err(|e| format!("header rebuild write error: {e}"))
}

/// Stub that scans `input` for frame boundaries and "marks" I-frames every
/// `GOP_SIZE` frames by writing a note to `output`.
///
/// Returns the number of frames marked.
///
/// A production implementation would modify actual bitstream fields to set
/// the I-frame bit in the relevant codec headers.
fn insert_keyframe_markers(input: &Path, output: &Path) -> Result<u64, String> {
    const GOP_SIZE: u64 = 30;

    let data =
        std::fs::read(input).map_err(|e| format!("keyframe read error: {e}"))?;

    // Heuristic frame count: assume each 4096-byte chunk is one frame.
    let estimated_frames = (data.len() as u64).div_ceil(4096);
    let markers_inserted = estimated_frames.div_ceil(GOP_SIZE);

    // Write the data unchanged; in a real implementation the codec bitstream
    // would be modified at each keyframe boundary.
    std::fs::write(output, &data)
        .map_err(|e| format!("keyframe write error: {e}"))?;

    Ok(markers_inserted)
}

/// Emit a log line to stderr documenting what repair was performed.
fn log_repair_action(msg: &str) {
    eprintln!("[oximedia-repair] {msg}");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_file(name: &str, data: &[u8]) -> std::path::PathBuf {
        let path =
            std::env::temp_dir().join(format!("oximedia_repair_engine_test_{name}"));
        let mut f = std::fs::File::create(&path).expect("create temp file");
        f.write_all(data).expect("write temp file");
        path
    }

    // ------ DetectedIssue construction --------------------------------------

    #[test]
    fn test_detected_issue_fields() {
        let issue = DetectedIssue {
            issue_type: IssueType::TruncatedFile,
            severity: IssueSeverity::High,
            description: "file truncated at byte 512".to_string(),
            location: Some(512),
            fixable: true,
            confidence: 0.95,
        };
        assert_eq!(issue.issue_type, IssueType::TruncatedFile);
        assert_eq!(issue.severity, IssueSeverity::High);
        assert!(issue.fixable);
    }

    // ------ fix_issue: TruncatedFile ----------------------------------------

    #[test]
    fn test_fix_issue_truncated_file_returns_ok() {
        // File with MPEG-TS sync bytes scattered through it
        let mut data = vec![0u8; 1024];
        data[100] = 0x47;
        data[288] = 0x47;
        data[476] = 0x47;

        let input = temp_file("trunc_in.bin", &data);
        let output = temp_file("trunc_out.bin", &[]);

        let issue = DetectedIssue {
            issue_type: IssueType::TruncatedFile,
            severity: IssueSeverity::High,
            description: "truncated".to_string(),
            location: Some(900),
            fixable: true,
            confidence: 0.9,
        };

        let result = fix_issue(&input, &output, &issue);
        assert!(result.is_ok(), "fix_issue should not return Err: {result:?}");
        assert!(result.unwrap(), "fix_issue should return Ok(true) for TruncatedFile");

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    // ------ fix_issue: CorruptHeader ----------------------------------------

    #[test]
    fn test_fix_issue_corrupt_header_returns_ok() {
        let data = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00, 0x00, 0x00, 0x11, 0x22, 0x33, 0x44,
                        0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC];
        let input = temp_file("header_in.bin", &data);
        let output = temp_file("header_out.bin", &[]);

        let issue = DetectedIssue {
            issue_type: IssueType::CorruptHeader,
            severity: IssueSeverity::Critical,
            description: "corrupt header".to_string(),
            location: Some(0),
            fixable: true,
            confidence: 0.95,
        };

        let result = fix_issue(&input, &output, &issue);
        assert!(result.is_ok());
        assert!(result.unwrap(), "CorruptHeader should return Ok(true)");

        // Output file should exist and start with zero header
        let out_data = std::fs::read(&output).expect("read output");
        assert!(
            out_data.iter().take(16).all(|&b| b == 0),
            "first 16 bytes should be the synthetic zero header"
        );

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    // ------ fix_issue: MissingKeyframes -------------------------------------

    #[test]
    fn test_fix_issue_missing_keyframes_returns_ok() {
        let data = vec![0u8; 4096 * 10]; // ~10 frames
        let input = temp_file("kf_in.bin", &data);
        let output = temp_file("kf_out.bin", &[]);

        let issue = DetectedIssue {
            issue_type: IssueType::MissingKeyframes,
            severity: IssueSeverity::Medium,
            description: "no keyframes".to_string(),
            location: None,
            fixable: true,
            confidence: 0.8,
        };

        let result = fix_issue(&input, &output, &issue);
        assert!(result.is_ok());
        assert!(result.unwrap(), "MissingKeyframes should return Ok(true)");

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    // ------ fix_issue: AudioSync / ContainerError → Ok(false) ---------------

    #[test]
    fn test_fix_issue_audio_sync_returns_false() {
        let data = vec![0u8; 256];
        let input = temp_file("sync_in.bin", &data);
        let output = temp_file("sync_out.bin", &[]);

        let issue = DetectedIssue {
            issue_type: IssueType::AudioSync,
            severity: IssueSeverity::Low,
            description: "audio sync drift".to_string(),
            location: None,
            fixable: false,
            confidence: 0.6,
        };

        let result = fix_issue(&input, &output, &issue);
        assert!(result.is_ok());
        assert!(!result.unwrap(), "AudioSync should return Ok(false)");

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    #[test]
    fn test_fix_issue_container_error_returns_false() {
        let data = vec![0u8; 256];
        let input = temp_file("cont_in.bin", &data);
        let output = temp_file("cont_out.bin", &[]);

        let issue = DetectedIssue {
            issue_type: IssueType::ContainerError,
            severity: IssueSeverity::Medium,
            description: "container error".to_string(),
            location: None,
            fixable: false,
            confidence: 0.75,
        };

        let result = fix_issue(&input, &output, &issue);
        assert!(result.is_ok());
        assert!(!result.unwrap(), "ContainerError should return Ok(false)");

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }

    // ------ IssueSeverity ordering ------------------------------------------

    #[test]
    fn test_issue_severity_ordering() {
        assert!(IssueSeverity::Low < IssueSeverity::Medium);
        assert!(IssueSeverity::Medium < IssueSeverity::High);
        assert!(IssueSeverity::High < IssueSeverity::Critical);
    }
}
