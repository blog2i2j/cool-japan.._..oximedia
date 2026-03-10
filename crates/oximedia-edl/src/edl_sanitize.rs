#![allow(dead_code)]
//! EDL text sanitization and normalization utilities.
//!
//! This module handles cleaning raw EDL text before parsing: removing
//! invalid characters, normalizing line endings, fixing common formatting
//! issues, and validating structural integrity of the text.

use crate::error::{EdlError, EdlResult};

/// Options controlling sanitization behaviour.
#[derive(Debug, Clone)]
pub struct SanitizeOptions {
    /// Remove blank lines (keep at most one between events).
    pub collapse_blank_lines: bool,
    /// Trim trailing whitespace from every line.
    pub trim_trailing: bool,
    /// Normalize all line endings to LF.
    pub normalize_line_endings: bool,
    /// Strip non-ASCII characters.
    pub strip_non_ascii: bool,
    /// Upper-case the FCM line and track-type tokens.
    pub uppercase_keywords: bool,
    /// Maximum allowed line length (0 = unlimited).
    pub max_line_length: usize,
}

impl Default for SanitizeOptions {
    fn default() -> Self {
        Self {
            collapse_blank_lines: true,
            trim_trailing: true,
            normalize_line_endings: true,
            strip_non_ascii: false,
            uppercase_keywords: true,
            max_line_length: 0,
        }
    }
}

/// Result of sanitization.
#[derive(Debug, Clone)]
pub struct SanitizeReport {
    /// Number of lines processed.
    pub lines_processed: usize,
    /// Number of lines modified.
    pub lines_modified: usize,
    /// Number of blank lines removed.
    pub blank_lines_removed: usize,
    /// Number of non-ASCII characters stripped.
    pub non_ascii_stripped: usize,
    /// Number of lines truncated.
    pub lines_truncated: usize,
}

/// Sanitize raw EDL text according to the given options.
///
/// # Errors
///
/// Returns an error if the input is empty after sanitization.
pub fn sanitize_edl(input: &str, opts: &SanitizeOptions) -> EdlResult<(String, SanitizeReport)> {
    let mut report = SanitizeReport {
        lines_processed: 0,
        lines_modified: 0,
        blank_lines_removed: 0,
        non_ascii_stripped: 0,
        lines_truncated: 0,
    };

    // Step 1: normalize line endings
    let text = if opts.normalize_line_endings {
        input.replace("\r\n", "\n").replace('\r', "\n")
    } else {
        input.to_string()
    };

    let mut output_lines: Vec<String> = Vec::new();
    let mut prev_blank = false;

    for line in text.split('\n') {
        report.lines_processed += 1;
        let mut l = line.to_string();
        let original = l.clone();

        // Trim trailing whitespace
        if opts.trim_trailing {
            let trimmed = l.trim_end().to_string();
            if trimmed.len() != l.len() {
                report.lines_modified += 1;
            }
            l = trimmed;
        }

        // Strip non-ASCII
        if opts.strip_non_ascii {
            let before_len = l.len();
            l = l.chars().filter(|c| c.is_ascii()).collect();
            let diff = before_len - l.len();
            if diff > 0 {
                report.non_ascii_stripped += diff;
                report.lines_modified += 1;
            }
        }

        // Uppercase keywords
        if opts.uppercase_keywords {
            l = uppercase_edl_keywords(&l, &original, &mut report);
        }

        // Max line length
        if opts.max_line_length > 0 && l.len() > opts.max_line_length {
            l.truncate(opts.max_line_length);
            report.lines_truncated += 1;
        }

        // Collapse blank lines
        let is_blank = l.trim().is_empty();
        if is_blank && opts.collapse_blank_lines {
            if prev_blank {
                report.blank_lines_removed += 1;
                continue;
            }
            prev_blank = true;
        } else {
            prev_blank = false;
        }

        output_lines.push(l);
    }

    // Remove trailing empty lines
    while output_lines.last().map_or(false, |l| l.trim().is_empty()) {
        output_lines.pop();
    }

    let result = output_lines.join("\n");
    if result.trim().is_empty() {
        return Err(EdlError::validation("EDL is empty after sanitization"));
    }

    Ok((result, report))
}

/// Uppercase known EDL keywords in a line.
fn uppercase_edl_keywords(
    line: &str,
    _original: &str,
    _report: &mut SanitizeReport,
) -> String {
    let trimmed = line.trim_start();
    // FCM line
    if trimmed.starts_with("fcm:") || trimmed.starts_with("FCM:") {
        return line.to_uppercase();
    }
    // TITLE line
    if let Some(rest) = trimmed.strip_prefix("title:").or_else(|| trimmed.strip_prefix("TITLE:")) {
        return format!("TITLE:{rest}");
    }
    line.to_string()
}

/// Validate that the text looks like an EDL structurally (has event lines).
///
/// # Errors
///
/// Returns an error if no event lines are found.
pub fn validate_edl_structure(text: &str) -> EdlResult<usize> {
    let mut event_count = 0;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('*') || trimmed.starts_with("TITLE")
            || trimmed.starts_with("FCM")
        {
            continue;
        }
        // Event lines start with a number
        if trimmed.chars().next().map_or(false, |c| c.is_ascii_digit()) {
            event_count += 1;
        }
    }
    if event_count == 0 {
        return Err(EdlError::validation(
            "No event lines found in EDL structure",
        ));
    }
    Ok(event_count)
}

/// Normalise reel names: strip quotes and limit to 8 characters (CMX convention).
#[must_use]
pub fn normalize_reel_name(name: &str) -> String {
    let stripped = name.trim().trim_matches('"').trim_matches('\'');
    if stripped.len() > 8 {
        stripped[..8].to_uppercase()
    } else {
        stripped.to_uppercase()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_basic() {
        let input = "TITLE: Test\nFCM: NON-DROP FRAME\n\n001  AX  V  C  01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n";
        let (out, report) = sanitize_edl(input, &SanitizeOptions::default()).expect("operation should succeed");
        assert!(out.contains("TITLE:"));
        assert!(report.lines_processed > 0);
    }

    #[test]
    fn test_sanitize_crlf() {
        let input = "TITLE: Test\r\nFCM: DROP FRAME\r\n001  AX  V  C  01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\r\n";
        let (out, _) = sanitize_edl(input, &SanitizeOptions::default()).expect("operation should succeed");
        assert!(!out.contains('\r'));
    }

    #[test]
    fn test_sanitize_collapse_blanks() {
        let input = "TITLE: A\n\n\n\n001  AX  V  C  01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n";
        let (out, report) = sanitize_edl(input, &SanitizeOptions::default()).expect("operation should succeed");
        assert!(report.blank_lines_removed > 0);
        // At most one blank line in a row
        assert!(!out.contains("\n\n\n"));
    }

    #[test]
    fn test_sanitize_empty_result() {
        let input = "   \n  \n  ";
        let result = sanitize_edl(input, &SanitizeOptions::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_sanitize_strip_non_ascii() {
        let mut opts = SanitizeOptions::default();
        opts.strip_non_ascii = true;
        let input = "TITLE: T\u{00e9}st\n001  AX  V  C  01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n";
        let (out, report) = sanitize_edl(input, &opts).expect("operation should succeed");
        assert!(!out.contains('\u{00e9}'));
        assert!(report.non_ascii_stripped > 0);
    }

    #[test]
    fn test_sanitize_max_line_length() {
        let mut opts = SanitizeOptions::default();
        opts.max_line_length = 20;
        let input = "TITLE: A very very very long title that exceeds maximum length\n001  AX  V  C  01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n";
        let (out, report) = sanitize_edl(input, &opts).expect("operation should succeed");
        for line in out.lines() {
            assert!(line.len() <= 20);
        }
        assert!(report.lines_truncated > 0);
    }

    #[test]
    fn test_sanitize_trim_trailing() {
        let input = "TITLE: Test   \n001  AX  V  C  01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00   \n";
        let (out, _) = sanitize_edl(input, &SanitizeOptions::default()).expect("operation should succeed");
        for line in out.lines() {
            assert_eq!(line, line.trim_end());
        }
    }

    #[test]
    fn test_validate_structure_valid() {
        let text = "TITLE: Test\nFCM: NON-DROP FRAME\n001  AX  V  C  01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n";
        let count = validate_edl_structure(text).expect("operation should succeed");
        assert_eq!(count, 1);
    }

    #[test]
    fn test_validate_structure_no_events() {
        let text = "TITLE: Test\nFCM: NON-DROP FRAME\n* comment only\n";
        let result = validate_edl_structure(text);
        assert!(result.is_err());
    }

    #[test]
    fn test_normalize_reel_name_short() {
        assert_eq!(normalize_reel_name("ax"), "AX");
    }

    #[test]
    fn test_normalize_reel_name_long() {
        assert_eq!(normalize_reel_name("VeryLongReelName"), "VERYLONG");
    }

    #[test]
    fn test_normalize_reel_name_quoted() {
        assert_eq!(normalize_reel_name("\"REEL01\""), "REEL01");
    }

    #[test]
    fn test_uppercase_fcm_line() {
        let input = "fcm: drop frame\n001  AX  V  C  01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00\n";
        let (out, _) = sanitize_edl(input, &SanitizeOptions::default()).expect("operation should succeed");
        assert!(out.contains("FCM: DROP FRAME"));
    }
}
