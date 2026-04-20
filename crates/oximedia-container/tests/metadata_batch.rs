//! Integration tests for multi-file batch metadata operations via public API.
//!
//! These tests exercise the public re-exports `BatchMetadataUpdate` and `BatchResult`
//! from the `oximedia_container` crate root.  Because actual file I/O against real
//! media files is environment-sensitive, most tests exercise the builder contract,
//! the error-collection semantics, and the report formatting using non-existent
//! paths (which produce deterministic failures).

use oximedia_container::{BatchMetadataUpdate, BatchResult};
use std::path::PathBuf;

fn tmp_str(name: &str) -> String {
    std::env::temp_dir()
        .join(format!("oximedia-container-metadata-batch-{name}"))
        .to_string_lossy()
        .into_owned()
}

// ─── BatchResult public API ───────────────────────────────────────────────────

#[test]
fn batch_result_default_is_empty() {
    let r = BatchResult::default();
    assert!(r.ok.is_empty());
    assert!(r.failed.is_empty());
    assert_eq!(r.total(), 0);
    assert!(r.all_succeeded());
}

#[test]
fn batch_result_total_is_sum_of_ok_and_failed() {
    let mut r = BatchResult::default();
    r.ok.push(PathBuf::from("a.flac"));
    r.ok.push(PathBuf::from("b.flac"));
    // Push a fake failure directly (we don't construct BatchError outside the module,
    // so we rely on apply() to produce failures for us here — tested separately).
    assert_eq!(r.total(), 2);
    assert!(r.all_succeeded());
}

#[test]
fn batch_result_report_success_contains_count() {
    let mut r = BatchResult::default();
    r.ok.push(PathBuf::from("track01.flac"));
    r.ok.push(PathBuf::from("track02.flac"));
    r.ok.push(PathBuf::from("track03.flac"));
    let report = r.into_report();
    assert!(report.contains("3"), "report must mention count 3");
    assert!(
        report.contains("successfully"),
        "report must say 'successfully'"
    );
}

// ─── BatchMetadataUpdate builder ─────────────────────────────────────────────

#[test]
fn builder_new_is_default() {
    let b = BatchMetadataUpdate::new();
    // apply() on an empty builder should produce a zero-item result.
    let r = b.apply();
    assert_eq!(r.total(), 0);
    assert!(r.all_succeeded());
}

#[test]
fn builder_add_file_then_no_tags_fails() {
    // A file path that definitely does not exist should produce a failure.
    let result = BatchMetadataUpdate::new()
        .add_file(tmp_str("int_test_nonexistent_7x9k.flac"))
        .set_tag("TITLE", "Integration Test")
        .apply();

    assert_eq!(result.ok.len(), 0);
    assert_eq!(result.failed.len(), 1, "one non-existent file must fail");
}

#[test]
fn builder_multiple_nonexistent_files_all_fail() {
    let result = BatchMetadataUpdate::new()
        .add_file(tmp_str("int_missing_a.flac"))
        .add_file(tmp_str("int_missing_b.flac"))
        .add_file(tmp_str("int_missing_c.ogg"))
        .set_tag("ALBUM", "Test Album")
        .set_tag("DATE", "2026")
        .apply();

    assert_eq!(result.ok.len(), 0, "no files should succeed");
    assert_eq!(result.failed.len(), 3, "all three missing files must fail");
    assert!(!result.all_succeeded());
}

#[test]
fn builder_report_for_partial_failure_mentions_paths() {
    let result = BatchMetadataUpdate::new()
        .add_file(tmp_str("int_report_missing_x.wav"))
        .add_file(tmp_str("int_report_missing_y.wav"))
        .set_tag("GENRE", "Electronica")
        .apply();

    assert_eq!(result.failed.len(), 2);
    let report = result.into_report();
    // The report should mention both file names.
    assert!(
        report.contains("int_report_missing_x.wav"),
        "report must name the first failed path; got: {report}"
    );
    assert!(
        report.contains("int_report_missing_y.wav"),
        "report must name the second failed path; got: {report}"
    );
}

// ─── set_tag chaining ─────────────────────────────────────────────────────────

#[test]
fn set_tag_multiple_keys_are_recorded() {
    // We can't inspect internal state after apply() starts, but we can confirm the
    // builder API is chainable and the apply() call does not panic.
    let result = BatchMetadataUpdate::new()
        .set_tag("TITLE", "Hello")
        .set_tag("ARTIST", "World")
        .set_tag("DATE", "2026")
        // No files — result should be empty but clean.
        .apply();

    assert_eq!(result.total(), 0);
    assert!(result.all_succeeded());
}

// ─── copy_from with missing source ────────────────────────────────────────────

#[test]
fn copy_from_nonexistent_source_fails_all_targets() {
    // When the copy_from source does not exist, every target should fail.
    let result = BatchMetadataUpdate::new()
        .copy_from(
            tmp_str("int_copy_source_missing.flac"),
            vec!["TITLE".into(), "ARTIST".into()],
        )
        .add_file(tmp_str("int_copy_target_a.flac"))
        .add_file(tmp_str("int_copy_target_b.flac"))
        .apply();

    assert_eq!(
        result.ok.len(),
        0,
        "no targets should succeed when source is missing"
    );
    assert_eq!(result.failed.len(), 2, "both targets must fail");
}

// ─── apply with no files ──────────────────────────────────────────────────────

#[test]
fn apply_no_files_returns_zero_total() {
    let result = BatchMetadataUpdate::new()
        .set_tag("COMMENT", "Unused tag, no files registered")
        .apply();

    assert_eq!(result.total(), 0);
    let report = result.into_report();
    // The report should still be valid (no panic).
    assert!(!report.is_empty());
}

// ─── all_succeeded semantics ──────────────────────────────────────────────────

#[test]
fn all_succeeded_true_when_no_files() {
    let result = BatchMetadataUpdate::new().apply();
    assert!(
        result.all_succeeded(),
        "zero files processed = trivially succeeded"
    );
}

#[test]
fn all_succeeded_false_when_any_fail() {
    let result = BatchMetadataUpdate::new()
        .add_file(tmp_str("int_any_fail.flac"))
        .apply();
    assert!(
        !result.all_succeeded(),
        "one failure means all_succeeded is false"
    );
}
