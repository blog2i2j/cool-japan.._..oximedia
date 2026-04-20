//! Duration and seek-point parsing for FFmpeg-style `-ss`, `-to`, and `-t` arguments.
//!
//! ## Supported Duration Formats
//!
//! | Format | Example | Meaning |
//! |--------|---------|---------|
//! | `hh:mm:ss.mmm` | `"01:23:45.678"` | 1 h 23 min 45.678 s |
//! | `mm:ss.mmm` | `"23:45.678"` | 23 min 45.678 s |
//! | `ss.mmm` | `"45.678"` | 45.678 s |
//! | Integer seconds | `"120"` | 120 s |
//! | `Nh` | `"2h"` | 2 hours |
//! | `Nm` | `"90m"` | 90 minutes |
//! | `Ns` | `"300s"` | 300 seconds |
//!
//! ## Usage
//!
//! ```rust
//! use oximedia_compat_ffmpeg::seek::{parse_duration, check_seek_args};
//! use std::time::Duration;
//!
//! let d = parse_duration("01:23:45.500").unwrap();
//! assert_eq!(d, Duration::from_millis(5025_500));
//!
//! let d2 = parse_duration("2h").unwrap();
//! assert_eq!(d2, Duration::from_secs(7200));
//! ```

use std::time::Duration;

use crate::arg_parser::OutputSpec;

/// Error type for seek / duration parsing.
#[derive(Debug, thiserror::Error)]
pub enum SeekError {
    /// The duration string did not match any recognised format.
    #[error("unrecognised duration format: '{0}'")]
    UnrecognisedFormat(String),

    /// A numeric component overflowed or was out of range.
    #[error("duration component out of range in '{0}': {1}")]
    OutOfRange(String, String),

    /// Both `-to` and `-t` were specified, which is a conflict.
    #[error("conflicting arguments: '-to' and '-t' must not both be specified")]
    ConflictingArgs,
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a duration string in one of the FFmpeg-accepted formats into a
/// [`std::time::Duration`].
///
/// Supported formats (see module doc for full table):
/// - `hh:mm:ss[.fractional]`
/// - `mm:ss[.fractional]`
/// - `ss[.fractional]`
/// - `<integer>` (seconds)
/// - `<N>h`, `<N>m`, `<N>s` (unit suffixes; `N` may be a float)
///
/// The function is intentionally strict: unknown trailing characters are an error.
pub fn parse_duration(s: &str) -> Result<Duration, SeekError> {
    let s = s.trim();

    if s.is_empty() {
        return Err(SeekError::UnrecognisedFormat(s.to_string()));
    }

    // ── Unit suffix forms: Nh / Nm / Ns ────────────────────────────────────
    if let Some(d) = try_parse_unit_suffix(s) {
        return d;
    }

    // ── Colon-separated forms: [hh:]mm:ss[.frac] ───────────────────────────
    if s.contains(':') {
        return parse_colon_form(s);
    }

    // ── Plain numeric (integer or float seconds) ────────────────────────────
    parse_plain_seconds(s)
}

/// Validate the seek arguments in an [`OutputSpec`], returning an error
/// if `-to` and `-t` are both specified (they are mutually exclusive in FFmpeg).
///
/// `-to` is stored in `OutputSpec::extra_args` as `("-to", value)`.
/// `-t` / `duration` is stored in `OutputSpec::duration`.
pub fn check_seek_args(output: &OutputSpec) -> Result<(), SeekError> {
    let has_t = output.duration.is_some();
    let has_to = output.extra_args.iter().any(|(k, _)| k == "-to");

    if has_t && has_to {
        return Err(SeekError::ConflictingArgs);
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Attempt to parse a unit-suffix form (`Nh`, `Nm`, `Ns`).
///
/// Returns `None` if the string does not end with `h`, `m`, or `s`.
/// Returns `Some(Result<…>)` if the suffix matched (even if parsing failed).
fn try_parse_unit_suffix(s: &str) -> Option<Result<Duration, SeekError>> {
    let last = s.chars().last()?;
    let unit = match last {
        'h' => Some(3600.0_f64),
        'm' => Some(60.0_f64),
        's' => Some(1.0_f64),
        _ => None,
    }?;

    let number_part = &s[..s.len() - 1];
    let n: f64 = match number_part.parse() {
        Ok(v) => v,
        Err(_) => {
            return Some(Err(SeekError::UnrecognisedFormat(s.to_string())));
        }
    };

    if n < 0.0 {
        return Some(Err(SeekError::OutOfRange(
            s.to_string(),
            "negative duration".to_string(),
        )));
    }

    let total_secs = n * unit;
    Some(Ok(Duration::from_secs_f64(total_secs)))
}

/// Parse a colon-separated duration: `hh:mm:ss[.frac]` or `mm:ss[.frac]`.
fn parse_colon_form(s: &str) -> Result<Duration, SeekError> {
    let parts: Vec<&str> = s.split(':').collect();

    match parts.len() {
        2 => {
            // mm:ss[.frac]
            let mm = parse_integer_component(s, parts[0])?;
            let (ss, frac_nanos) = parse_seconds_component(s, parts[1])?;
            if ss >= 60 {
                return Err(SeekError::OutOfRange(
                    s.to_string(),
                    format!("seconds component {} >= 60", ss),
                ));
            }
            let total_nanos = (mm as u64 * 60 + ss as u64) * 1_000_000_000 + frac_nanos;
            Ok(Duration::from_nanos(total_nanos))
        }
        3 => {
            // hh:mm:ss[.frac]
            let hh = parse_integer_component(s, parts[0])?;
            let mm = parse_integer_component(s, parts[1])?;
            let (ss, frac_nanos) = parse_seconds_component(s, parts[2])?;
            if mm >= 60 {
                return Err(SeekError::OutOfRange(
                    s.to_string(),
                    format!("minutes component {} >= 60", mm),
                ));
            }
            if ss >= 60 {
                return Err(SeekError::OutOfRange(
                    s.to_string(),
                    format!("seconds component {} >= 60", ss),
                ));
            }
            let total_nanos =
                (hh as u64 * 3600 + mm as u64 * 60 + ss as u64) * 1_000_000_000 + frac_nanos;
            Ok(Duration::from_nanos(total_nanos))
        }
        _ => Err(SeekError::UnrecognisedFormat(s.to_string())),
    }
}

/// Parse a plain-seconds string: `"45"` or `"45.678"`.
fn parse_plain_seconds(s: &str) -> Result<Duration, SeekError> {
    let v: f64 = s
        .parse()
        .map_err(|_| SeekError::UnrecognisedFormat(s.to_string()))?;
    if v < 0.0 {
        return Err(SeekError::OutOfRange(
            s.to_string(),
            "negative duration".to_string(),
        ));
    }
    Ok(Duration::from_secs_f64(v))
}

/// Parse a non-negative integer component from a colon-split segment.
fn parse_integer_component(full: &str, part: &str) -> Result<u32, SeekError> {
    part.parse::<u32>().map_err(|_| {
        SeekError::OutOfRange(full.to_string(), format!("'{}' is not an integer", part))
    })
}

/// Parse the seconds component, which may include a fractional part: `"45"` or `"45.678"`.
///
/// Returns `(integer_seconds, fractional_nanoseconds)`.
fn parse_seconds_component(full: &str, part: &str) -> Result<(u32, u64), SeekError> {
    if let Some(dot_pos) = part.find('.') {
        let int_str = &part[..dot_pos];
        let frac_str = &part[dot_pos + 1..];

        let secs: u32 = int_str.parse().map_err(|_| {
            SeekError::OutOfRange(full.to_string(), format!("'{}' is not an integer", int_str))
        })?;

        // Normalise fractional part to nanoseconds (9 decimal digits).
        let frac_nanos = parse_fractional_nanos(frac_str).map_err(|_| {
            SeekError::OutOfRange(
                full.to_string(),
                format!("'{}' is not a valid fractional seconds", frac_str),
            )
        })?;

        Ok((secs, frac_nanos))
    } else {
        let secs: u32 = part.parse().map_err(|_| {
            SeekError::OutOfRange(full.to_string(), format!("'{}' is not an integer", part))
        })?;
        Ok((secs, 0))
    }
}

/// Convert a decimal-fraction string (after the `.`) to nanoseconds.
///
/// `"5"` → 500_000_000 ns (0.5 s), `"500"` → 500_000_000 ns, `"678"` → 678_000_000 ns.
fn parse_fractional_nanos(frac: &str) -> Result<u64, ()> {
    if frac.is_empty() {
        return Ok(0);
    }
    // Ensure all characters are digits.
    if !frac.bytes().all(|b| b.is_ascii_digit()) {
        return Err(());
    }
    // Pad or truncate to 9 digits.
    let digits: String = if frac.len() < 9 {
        format!("{:0<9}", frac)
    } else {
        frac[..9].to_string()
    };
    digits.parse::<u64>().map_err(|_| ())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hh_mm_ss_millis() {
        let d = parse_duration("01:23:45.678").expect("parse");
        // 1*3600 + 23*60 + 45 = 5025 seconds + 0.678
        let expected = Duration::from_millis(5025_678);
        assert_eq!(d, expected);
    }

    #[test]
    fn test_hh_mm_ss_no_frac() {
        let d = parse_duration("00:01:30").expect("parse");
        assert_eq!(d, Duration::from_secs(90));
    }

    #[test]
    fn test_mm_ss() {
        let d = parse_duration("05:30").expect("parse");
        assert_eq!(d, Duration::from_secs(330));
    }

    #[test]
    fn test_mm_ss_frac() {
        let d = parse_duration("01:00.500").expect("parse");
        assert_eq!(d, Duration::from_millis(60_500));
    }

    #[test]
    fn test_plain_integer() {
        let d = parse_duration("120").expect("parse");
        assert_eq!(d, Duration::from_secs(120));
    }

    #[test]
    fn test_plain_float() {
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
    fn test_float_seconds_suffix() {
        let d = parse_duration("1.5s").expect("parse");
        assert_eq!(d, Duration::from_millis(1500));
    }

    #[test]
    fn test_zero() {
        let d = parse_duration("0").expect("parse");
        assert_eq!(d, Duration::ZERO);
    }

    #[test]
    fn test_invalid_format() {
        assert!(parse_duration("abc").is_err());
        assert!(parse_duration("").is_err());
        assert!(parse_duration("1:2:3:4").is_err());
    }

    #[test]
    fn test_check_seek_no_conflict() {
        let spec = OutputSpec {
            duration: Some("30".to_string()),
            ..OutputSpec::default()
        };
        assert!(check_seek_args(&spec).is_ok());
    }

    #[test]
    fn test_check_seek_conflict() {
        let mut spec = OutputSpec {
            duration: Some("30".to_string()),
            ..OutputSpec::default()
        };
        spec.extra_args.push(("-to".to_string(), "60".to_string()));
        let err = check_seek_args(&spec).expect_err("should conflict");
        assert!(matches!(err, SeekError::ConflictingArgs));
    }

    #[test]
    fn test_check_seek_only_to() {
        let mut spec = OutputSpec::default();
        spec.extra_args.push(("-to".to_string(), "60".to_string()));
        assert!(check_seek_args(&spec).is_ok());
    }

    #[test]
    fn test_fractional_nanos_precision() {
        // "1" after the decimal point → 100_000_000 ns = 0.1 s
        let d = parse_duration("0:00:00.1").expect("parse");
        assert_eq!(d, Duration::from_millis(100));
    }
}
