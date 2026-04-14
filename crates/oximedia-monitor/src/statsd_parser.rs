//! StatsD wire-format parser (free-function API).
//!
//! Parses individual StatsD metric lines or batch payloads without requiring a
//! UDP socket.  All parsing is pure string processing — suitable for unit
//! testing and for feeding metrics from any byte source (file, pipe, queue).
//!
//! # Wire format
//!
//! ```text
//! metric.name:value|type[|@sample_rate][|#key:val,key2:val2]
//! ```
//!
//! Supported type codes: `c` (counter), `g` (gauge), `ms` (timer),
//! `s` (set), `h` (histogram).
//!
//! DogStatsD tag syntax is supported via the `|#key:value,...` segment.
//!
//! # Example
//!
//! ```rust
//! use oximedia_monitor::statsd_parser::{parse_statsd_line, StatsdMetricType};
//!
//! let m = parse_statsd_line("cpu.usage:42.5|g").expect("valid line");
//! assert_eq!(m.name, "cpu.usage");
//! assert!((m.value - 42.5).abs() < f64::EPSILON);
//! assert_eq!(m.metric_type, StatsdMetricType::Gauge);
//! ```

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// StatsD metric type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StatsdMetricType {
    /// Monotonically increasing counter: `c`.
    Counter,
    /// Instantaneous gauge value: `g`.
    Gauge,
    /// Timing sample (milliseconds): `ms`.
    Timer,
    /// Unique element set: `s`.
    Set,
    /// Histogram distribution: `h`.
    Histogram,
}

impl StatsdMetricType {
    /// Parse from a StatsD type symbol.
    ///
    /// # Errors
    ///
    /// Returns a `String` error when the symbol is unrecognised.
    fn from_symbol(s: &str) -> Result<Self, String> {
        match s {
            "c" => Ok(Self::Counter),
            "g" => Ok(Self::Gauge),
            "ms" => Ok(Self::Timer),
            "s" => Ok(Self::Set),
            "h" => Ok(Self::Histogram),
            other => Err(format!("unknown StatsD type symbol: '{other}'")),
        }
    }

    /// Return the canonical wire-format symbol.
    #[must_use]
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Counter => "c",
            Self::Gauge => "g",
            Self::Timer => "ms",
            Self::Set => "s",
            Self::Histogram => "h",
        }
    }
}

/// A successfully parsed StatsD metric.
///
/// Tags follow the DogStatsD convention: `|#key:value,key2:value2`.
/// Tags without a `:` separator are stored with an empty-string value.
///
/// For [`StatsdMetricType::Set`] metrics the wire-format value is an arbitrary
/// string (e.g. a user ID), not necessarily a number.  In that case `value`
/// holds `f64::NAN` and `set_value` holds the raw string.  For all other
/// metric types `set_value` is `None`.
#[derive(Debug, Clone)]
pub struct StatsdMetric {
    /// Metric name (the segment before `:`).
    pub name: String,
    /// Parsed numeric value (`f64::NAN` for Set metrics with non-numeric values).
    pub value: f64,
    /// Raw string value for Set metrics.  `None` for all other metric types.
    pub set_value: Option<String>,
    /// Metric type.
    pub metric_type: StatsdMetricType,
    /// Sampling rate, if present (e.g. `0.1` for 10% sampling).
    pub sample_rate: Option<f64>,
    /// DogStatsD tags parsed as `(key, value)` pairs.
    pub tags: Vec<(String, String)>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse a single StatsD wire-format line.
///
/// Returns `Ok(StatsdMetric)` on success or an `Err(String)` describing the
/// parse failure.
///
/// # Errors
///
/// - Empty or whitespace-only input.
/// - Missing `:` separator between name and value.
/// - Empty metric name.
/// - Non-numeric value field.
/// - Missing or unrecognised type symbol.
pub fn parse_statsd_line(line: &str) -> Result<StatsdMetric, String> {
    let line = line.trim();
    if line.is_empty() {
        return Err("empty StatsD line".to_string());
    }

    // ── Split name from the rest ────────────────────────────────────────────
    let colon = line
        .find(':')
        .ok_or_else(|| format!("missing ':' separator in StatsD line: '{line}'"))?;

    let name = line[..colon].trim();
    if name.is_empty() {
        return Err("StatsD metric name is empty".to_string());
    }
    let name = name.to_string();

    let rest = &line[colon + 1..];

    // ── Split on '|' ────────────────────────────────────────────────────────
    let segments: Vec<&str> = rest.split('|').collect();
    if segments.len() < 2 {
        return Err(format!(
            "StatsD line missing '|type' section after value: '{line}'"
        ));
    }

    // ── Type ─────────────────────────────────────────────────────────────────
    // Parse type before value so we can handle Set strings correctly.
    let metric_type = StatsdMetricType::from_symbol(segments[1].trim())?;

    // ── Value ────────────────────────────────────────────────────────────────
    let value_str = segments[0].trim();

    // Set metrics carry a string element (e.g. a user ID) rather than a
    // numeric value, so we attempt a numeric parse but fall back gracefully.
    let (value, set_value) = if metric_type == StatsdMetricType::Set {
        match value_str.parse::<f64>() {
            Ok(n) => (n, None),
            Err(_) => (f64::NAN, Some(value_str.to_string())),
        }
    } else {
        let n: f64 = value_str
            .parse()
            .map_err(|_| format!("invalid StatsD value '{value_str}' in line: '{line}'"))?;
        (n, None)
    };

    // ── Optional extensions ──────────────────────────────────────────────────
    let mut sample_rate: Option<f64> = None;
    let mut tags: Vec<(String, String)> = Vec::new();

    for seg in &segments[2..] {
        let seg = seg.trim();
        if let Some(rate_str) = seg.strip_prefix('@') {
            let parsed: f64 = rate_str
                .parse()
                .map_err(|_| format!("invalid sample rate '{rate_str}' in line: '{line}'"))?;
            sample_rate = Some(parsed.clamp(0.0, 1.0));
        } else if let Some(tag_str) = seg.strip_prefix('#') {
            for token in tag_str.split(',') {
                let token = token.trim();
                if token.is_empty() {
                    continue;
                }
                if let Some(sep) = token.find(':') {
                    tags.push((token[..sep].to_string(), token[sep + 1..].to_string()));
                } else {
                    tags.push((token.to_string(), String::new()));
                }
            }
        }
    }

    Ok(StatsdMetric {
        name,
        value,
        set_value,
        metric_type,
        sample_rate,
        tags,
    })
}

/// Parse multiple StatsD lines from a single string.
///
/// Blank lines and lines beginning with `#` are silently skipped.
/// Each non-blank line produces an entry in the returned `Vec`; parse errors
/// are preserved so callers can inspect or log them.
#[must_use]
pub fn parse_statsd_batch(input: &str) -> Vec<Result<StatsdMetric, String>> {
    input
        .lines()
        .filter(|l| {
            let trimmed = l.trim();
            !trimmed.is_empty() && !trimmed.starts_with('#')
        })
        .map(parse_statsd_line)
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_statsd_line ────────────────────────────────────────────────────

    #[test]
    fn test_counter_basic() {
        let m = parse_statsd_line("foo.bar:1|c").expect("valid counter");
        assert_eq!(m.name, "foo.bar");
        assert!((m.value - 1.0).abs() < f64::EPSILON);
        assert_eq!(m.metric_type, StatsdMetricType::Counter);
        assert!(m.sample_rate.is_none());
        assert!(m.tags.is_empty());
    }

    #[test]
    fn test_gauge_float() {
        let m = parse_statsd_line("cpu.usage:42.5|g").expect("valid gauge");
        assert_eq!(m.name, "cpu.usage");
        assert!((m.value - 42.5).abs() < f64::EPSILON);
        assert_eq!(m.metric_type, StatsdMetricType::Gauge);
    }

    #[test]
    fn test_timer() {
        let m = parse_statsd_line("req.latency:100|ms").expect("valid timer");
        assert_eq!(m.name, "req.latency");
        assert!((m.value - 100.0).abs() < f64::EPSILON);
        assert_eq!(m.metric_type, StatsdMetricType::Timer);
    }

    #[test]
    fn test_set() {
        let m = parse_statsd_line("users:alice|s").expect("valid set");
        assert_eq!(m.name, "users");
        assert_eq!(m.metric_type, StatsdMetricType::Set);
        // Set metrics carry a string element, not a numeric value.
        assert_eq!(m.set_value.as_deref(), Some("alice"));
        assert!(m.value.is_nan());
    }

    #[test]
    fn test_set_numeric_value() {
        // Set metrics can also carry numeric string values (user-count style).
        let m = parse_statsd_line("active_users:42|s").expect("valid numeric set");
        assert_eq!(m.metric_type, StatsdMetricType::Set);
        assert!((m.value - 42.0).abs() < f64::EPSILON);
        assert!(m.set_value.is_none());
    }

    #[test]
    fn test_histogram() {
        let m = parse_statsd_line("payload.bytes:512|h").expect("valid histogram");
        assert_eq!(m.metric_type, StatsdMetricType::Histogram);
        assert!((m.value - 512.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sample_rate() {
        let m = parse_statsd_line("hits:1|c|@0.5").expect("valid line with sample rate");
        assert_eq!(m.metric_type, StatsdMetricType::Counter);
        assert_eq!(m.sample_rate, Some(0.5));
    }

    #[test]
    fn test_dogstatsd_tags() {
        let m = parse_statsd_line("req:1|c|#env:prod,service:api").expect("valid DogStatsD line");
        assert_eq!(m.tags.len(), 2);
        assert!(m.tags.iter().any(|(k, v)| k == "env" && v == "prod"));
        assert!(m.tags.iter().any(|(k, v)| k == "service" && v == "api"));
    }

    #[test]
    fn test_combined_sample_rate_and_tags() {
        let m =
            parse_statsd_line("api.latency:250|ms|@0.1|#env:staging,version:2")
                .expect("valid combined line");
        assert_eq!(m.metric_type, StatsdMetricType::Timer);
        assert_eq!(m.sample_rate, Some(0.1));
        assert_eq!(m.tags.len(), 2);
        assert!(m.tags.iter().any(|(k, v)| k == "env" && v == "staging"));
        assert!(m.tags.iter().any(|(k, v)| k == "version" && v == "2"));
    }

    #[test]
    fn test_empty_name_errors() {
        let result = parse_statsd_line(":42|g");
        assert!(result.is_err(), "empty name should produce an error");
        let msg = result.unwrap_err();
        assert!(msg.contains("empty"), "error message should mention empty name");
    }

    #[test]
    fn test_missing_colon_errors() {
        let result = parse_statsd_line("no_colon_here");
        assert!(result.is_err(), "missing ':' should produce an error");
        let msg = result.unwrap_err();
        assert!(msg.contains("':'"), "error should mention missing separator");
    }

    #[test]
    fn test_invalid_value_errors() {
        let result = parse_statsd_line("metric:not_a_number|g");
        assert!(result.is_err(), "non-numeric value should error");
        let msg = result.unwrap_err();
        assert!(
            msg.contains("not_a_number") || msg.contains("invalid"),
            "error should describe the bad value"
        );
    }

    // ── parse_statsd_batch ───────────────────────────────────────────────────

    #[test]
    fn test_batch_basic() {
        let input = "cpu:75|g\nmem:50|g\n";
        let results = parse_statsd_batch(input);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.is_ok()));
    }

    #[test]
    fn test_batch_skips_blank_lines() {
        let input = "a:1|c\n\n\nb:2|c\n";
        let results = parse_statsd_batch(input);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_batch_skips_comment_lines() {
        let input = "# this is a comment\na:1|c\n# another comment\nb:2|g\n";
        let results = parse_statsd_batch(input);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.is_ok()));
    }

    #[test]
    fn test_batch_preserves_errors() {
        let input = "valid:1|c\nbad_line_no_colon\nother:2|g\n";
        let results = parse_statsd_batch(input);
        assert_eq!(results.len(), 3);
        assert!(results[0].is_ok());
        assert!(results[1].is_err());
        assert!(results[2].is_ok());
    }

    #[test]
    fn test_batch_empty_input() {
        let results = parse_statsd_batch("");
        assert!(results.is_empty());
    }
}
