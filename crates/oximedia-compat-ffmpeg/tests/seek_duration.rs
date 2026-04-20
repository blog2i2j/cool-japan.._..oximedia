//! Integration tests for duration parsing and seek-argument conflict detection.

use oximedia_compat_ffmpeg::arg_parser::OutputSpec;
use oximedia_compat_ffmpeg::seek::{check_seek_args, parse_duration, SeekError};
use std::time::Duration;

// ─────────────────────────────────────────────────────────────────────────────
// Duration parsing tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_hh_mm_ss_millis() {
    let d = parse_duration("01:23:45.678").expect("parse");
    assert_eq!(d, Duration::from_millis(5025_678));
}

#[test]
fn test_hh_mm_ss_no_frac() {
    let d = parse_duration("00:01:30").expect("parse");
    assert_eq!(d, Duration::from_secs(90));
}

#[test]
fn test_mm_ss_only() {
    let d = parse_duration("05:30").expect("parse");
    assert_eq!(d, Duration::from_secs(330));
}

#[test]
fn test_mm_ss_with_frac() {
    let d = parse_duration("01:00.500").expect("parse");
    assert_eq!(d, Duration::from_millis(60_500));
}

#[test]
fn test_plain_integer_seconds() {
    let d = parse_duration("120").expect("parse");
    assert_eq!(d, Duration::from_secs(120));
}

#[test]
fn test_plain_float_seconds() {
    let d = parse_duration("45.5").expect("parse");
    assert_eq!(d, Duration::from_millis(45_500));
}

#[test]
fn test_hours_suffix() {
    let d = parse_duration("2h").expect("parse");
    assert_eq!(d, Duration::from_secs(7200));
}

#[test]
fn test_minutes_suffix() {
    let d = parse_duration("90m").expect("parse");
    assert_eq!(d, Duration::from_secs(5400));
}

#[test]
fn test_seconds_suffix() {
    let d = parse_duration("300s").expect("parse");
    assert_eq!(d, Duration::from_secs(300));
}

#[test]
fn test_float_hours_suffix() {
    let d = parse_duration("1.5h").expect("parse");
    assert_eq!(d, Duration::from_secs(5400));
}

#[test]
fn test_zero_duration() {
    let d = parse_duration("0").expect("parse");
    assert_eq!(d, Duration::ZERO);
}

#[test]
fn test_fractional_precision_tenths() {
    // "0:00:00.1" → 100 ms
    let d = parse_duration("0:00:00.1").expect("parse");
    assert_eq!(d, Duration::from_millis(100));
}

// ─────────────────────────────────────────────────────────────────────────────
// Invalid format tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_empty_string() {
    assert!(parse_duration("").is_err());
}

#[test]
fn test_non_numeric() {
    assert!(parse_duration("abc").is_err());
}

#[test]
fn test_too_many_colons() {
    assert!(parse_duration("1:2:3:4").is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// Seek conflict detection
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_no_conflict_duration_only() {
    let spec = OutputSpec {
        duration: Some("30".to_string()),
        ..OutputSpec::default()
    };
    assert!(check_seek_args(&spec).is_ok());
}

#[test]
fn test_no_conflict_to_only() {
    let mut spec = OutputSpec::default();
    spec.extra_args.push(("-to".to_string(), "60".to_string()));
    assert!(check_seek_args(&spec).is_ok());
}

#[test]
fn test_no_conflict_neither() {
    assert!(check_seek_args(&OutputSpec::default()).is_ok());
}

#[test]
fn test_conflict_both_t_and_to() {
    let mut spec = OutputSpec {
        duration: Some("30".to_string()),
        ..OutputSpec::default()
    };
    spec.extra_args.push(("-to".to_string(), "60".to_string()));
    let err = check_seek_args(&spec).expect_err("should conflict");
    assert!(
        matches!(err, SeekError::ConflictingArgs),
        "expected ConflictingArgs, got {:?}",
        err
    );
}
