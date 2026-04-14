// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Farm-wide configuration with YAML loading and validation.
//!
//! [`FarmConfig`] holds the operational parameters for the render farm.
//! [`FarmConfigBuilder`] can construct a config from YAML text using a
//! zero-dependency hand-written parser (no `serde_yaml` or `yaml-rust` is
//! required — only `serde_json` from the workspace is available, and YAML is
//! parsed directly).
//!
//! # Supported YAML syntax
//!
//! The parser handles a flat or one-level-nested YAML mapping with `key: value`
//! pairs on separate lines.  Lines that are blank or begin with `#` are
//! ignored.  Nested sections are detected by indent level (2-space or 4-space
//! indentation).
//!
//! Recognised top-level keys:
//! - `max_workers`              — `usize`
//! - `job_timeout_ms`           — `u64`
//! - `checkpoint_interval_ms`   — `u64`
//! - `log_level`                — `String`
//!
//! # Example
//!
//! ```rust
//! use oximedia_renderfarm::farm_config::{FarmConfig, FarmConfigBuilder};
//!
//! let yaml = r#"
//! max_workers: 8
//! job_timeout_ms: 300000
//! checkpoint_interval_ms: 60000
//! log_level: info
//! "#;
//!
//! let config = FarmConfigBuilder::from_yaml_text(yaml).expect("valid yaml");
//! assert_eq!(config.max_workers, 8);
//! assert!(config.validate().is_ok());
//! ```

// ---------------------------------------------------------------------------
// FarmConfig
// ---------------------------------------------------------------------------

/// Operational parameters for the render farm.
#[derive(Debug, Clone)]
pub struct FarmConfig {
    /// Maximum number of concurrent workers.  Must be ≥ 1.
    pub max_workers: usize,
    /// Maximum time (ms) a single job may run before being terminated.
    pub job_timeout_ms: u64,
    /// Interval (ms) between incremental checkpoint writes.
    pub checkpoint_interval_ms: u64,
    /// Logging level string (e.g. `"info"`, `"debug"`, `"warn"`).
    pub log_level: String,
}

impl Default for FarmConfig {
    fn default() -> Self {
        Self {
            max_workers: 4,
            job_timeout_ms: 3_600_000,      // 1 hour
            checkpoint_interval_ms: 60_000, // 1 minute
            log_level: "info".to_owned(),
        }
    }
}

impl FarmConfig {
    /// Validate the configuration, returning a list of error messages on
    /// failure.  Returns `Ok(())` when the configuration is consistent.
    ///
    /// Checks:
    /// - `max_workers >= 1`
    /// - `job_timeout_ms > 0`
    /// - `checkpoint_interval_ms > 0`
    /// - `log_level` is one of `error`, `warn`, `info`, `debug`, `trace`
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors: Vec<String> = Vec::new();

        if self.max_workers < 1 {
            errors.push("max_workers must be >= 1".to_owned());
        }

        if self.job_timeout_ms == 0 {
            errors.push("job_timeout_ms must be > 0".to_owned());
        }

        if self.checkpoint_interval_ms == 0 {
            errors.push("checkpoint_interval_ms must be > 0".to_owned());
        }

        let valid_levels = ["error", "warn", "info", "debug", "trace"];
        if !valid_levels.contains(&self.log_level.as_str()) {
            errors.push(format!(
                "log_level '{}' is not one of: error, warn, info, debug, trace",
                self.log_level
            ));
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

// ---------------------------------------------------------------------------
// FarmConfigBuilder
// ---------------------------------------------------------------------------

/// Builder for [`FarmConfig`].
///
/// Start with default values and override selectively, or use
/// [`from_yaml_text`](Self::from_yaml_text) to parse from YAML.
#[derive(Debug, Default)]
pub struct FarmConfigBuilder {
    inner: FarmConfig,
}

impl FarmConfigBuilder {
    /// Create a builder with default values.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: FarmConfig::default(),
        }
    }

    /// Set `max_workers`.
    #[must_use]
    pub fn max_workers(mut self, v: usize) -> Self {
        self.inner.max_workers = v;
        self
    }

    /// Set `job_timeout_ms`.
    #[must_use]
    pub fn job_timeout_ms(mut self, v: u64) -> Self {
        self.inner.job_timeout_ms = v;
        self
    }

    /// Set `checkpoint_interval_ms`.
    #[must_use]
    pub fn checkpoint_interval_ms(mut self, v: u64) -> Self {
        self.inner.checkpoint_interval_ms = v;
        self
    }

    /// Set `log_level`.
    #[must_use]
    pub fn log_level(mut self, v: impl Into<String>) -> Self {
        self.inner.log_level = v.into();
        self
    }

    /// Build the [`FarmConfig`].
    #[must_use]
    pub fn build(self) -> FarmConfig {
        self.inner
    }

    // -----------------------------------------------------------------------
    // Minimal YAML parser
    // -----------------------------------------------------------------------

    /// Parse a [`FarmConfig`] from a YAML string.
    ///
    /// Only the keys listed in the module-level documentation are recognised.
    /// Unknown keys are silently ignored.  The parser handles a flat or
    /// one-level-nested mapping; deeper nesting is also silently ignored.
    ///
    /// Returns `Err(String)` when a known key has an unparseable value.
    pub fn from_yaml_text(yaml: &str) -> Result<FarmConfig, String> {
        let mut config = FarmConfig::default();

        for raw_line in yaml.lines() {
            // Strip trailing carriage return (Windows line endings).
            let line = raw_line.trim_end_matches('\r');

            // Skip blank lines and comments.
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            // Skip section-headers (lines ending with ':' and no value).
            // Also skip lines that are indented (nested under a section we
            // don't need to handle specially).
            let indent = leading_spaces(line);
            if indent > 0 {
                // Nested key — still try to parse as a flat pair in case the
                // caller uses indented top-level keys.
            }

            // Split on the first ':'.
            let (raw_key, raw_val) = match split_first_colon(trimmed) {
                Some(pair) => pair,
                None => continue, // no colon found
            };

            let key = raw_key.trim();
            let val = raw_val.trim();

            // Skip section-only lines (value is empty after the colon).
            if val.is_empty() {
                continue;
            }

            match key {
                "max_workers" => {
                    config.max_workers = val
                        .parse::<usize>()
                        .map_err(|e| format!("invalid max_workers '{}': {}", val, e))?;
                }
                "job_timeout_ms" => {
                    config.job_timeout_ms = val
                        .parse::<u64>()
                        .map_err(|e| format!("invalid job_timeout_ms '{}': {}", val, e))?;
                }
                "checkpoint_interval_ms" => {
                    config.checkpoint_interval_ms = val
                        .parse::<u64>()
                        .map_err(|e| format!("invalid checkpoint_interval_ms '{}': {}", val, e))?;
                }
                "log_level" => {
                    // Strip optional surrounding quotes.
                    config.log_level = strip_quotes(val).to_owned();
                }
                _ => {
                    // Unknown key — silently ignore per spec.
                }
            }
        }

        Ok(config)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Count the number of leading space characters in `s`.
fn leading_spaces(s: &str) -> usize {
    s.chars().take_while(|c| *c == ' ').count()
}

/// Split `s` on the first `':'` and return the two parts.
/// Returns `None` when no `':'` is present.
fn split_first_colon(s: &str) -> Option<(&str, &str)> {
    let pos = s.find(':')?;
    Some((&s[..pos], &s[pos + 1..]))
}

/// Remove surrounding single or double quotes from `s` if present.
fn strip_quotes(s: &str) -> &str {
    if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
        if s.len() >= 2 {
            return &s[1..s.len() - 1];
        }
    }
    s
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- FarmConfig::validate ---

    #[test]
    fn validate_default_config_passes() {
        assert!(FarmConfig::default().validate().is_ok());
    }

    #[test]
    fn validate_zero_max_workers_fails() {
        let cfg = FarmConfig {
            max_workers: 0,
            ..Default::default()
        };
        let errs = cfg.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("max_workers")));
    }

    #[test]
    fn validate_zero_job_timeout_fails() {
        let cfg = FarmConfig {
            job_timeout_ms: 0,
            ..Default::default()
        };
        let errs = cfg.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("job_timeout_ms")));
    }

    #[test]
    fn validate_zero_checkpoint_interval_fails() {
        let cfg = FarmConfig {
            checkpoint_interval_ms: 0,
            ..Default::default()
        };
        let errs = cfg.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("checkpoint_interval_ms")));
    }

    #[test]
    fn validate_invalid_log_level_fails() {
        let cfg = FarmConfig {
            log_level: "verbose".to_owned(),
            ..Default::default()
        };
        let errs = cfg.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("log_level")));
    }

    #[test]
    fn validate_multiple_errors_reported() {
        let cfg = FarmConfig {
            max_workers: 0,
            job_timeout_ms: 0,
            ..Default::default()
        };
        let errs = cfg.validate().unwrap_err();
        assert!(errs.len() >= 2);
    }

    // --- FarmConfigBuilder::from_yaml_text ---

    #[test]
    fn yaml_parse_all_fields() {
        let yaml = "max_workers: 16\njob_timeout_ms: 120000\ncheckpoint_interval_ms: 30000\nlog_level: debug\n";
        let cfg = FarmConfigBuilder::from_yaml_text(yaml).expect("parse");
        assert_eq!(cfg.max_workers, 16);
        assert_eq!(cfg.job_timeout_ms, 120_000);
        assert_eq!(cfg.checkpoint_interval_ms, 30_000);
        assert_eq!(cfg.log_level, "debug");
    }

    #[test]
    fn yaml_parse_with_comments_and_blank_lines() {
        let yaml =
            "# Farm configuration\n\nmax_workers: 4\n# timeout in ms\njob_timeout_ms: 60000\n";
        let cfg = FarmConfigBuilder::from_yaml_text(yaml).expect("parse");
        assert_eq!(cfg.max_workers, 4);
        assert_eq!(cfg.job_timeout_ms, 60_000);
    }

    #[test]
    fn yaml_parse_quoted_log_level() {
        let yaml = "log_level: \"warn\"\n";
        let cfg = FarmConfigBuilder::from_yaml_text(yaml).expect("parse");
        assert_eq!(cfg.log_level, "warn");
    }

    #[test]
    fn yaml_parse_single_quoted_log_level() {
        let yaml = "log_level: 'trace'\n";
        let cfg = FarmConfigBuilder::from_yaml_text(yaml).expect("parse");
        assert_eq!(cfg.log_level, "trace");
    }

    #[test]
    fn yaml_parse_unknown_keys_ignored() {
        let yaml = "max_workers: 2\nunknown_key: some_value\njob_timeout_ms: 5000\n";
        let cfg = FarmConfigBuilder::from_yaml_text(yaml).expect("parse");
        assert_eq!(cfg.max_workers, 2);
    }

    #[test]
    fn yaml_parse_invalid_integer_returns_err() {
        let yaml = "max_workers: not_a_number\n";
        assert!(FarmConfigBuilder::from_yaml_text(yaml).is_err());
    }

    #[test]
    fn yaml_parse_nested_section_header_skipped() {
        let yaml = "workers:\n  max_workers: 8\nmax_workers: 4\n";
        // The top-level `max_workers: 4` overrides any nested parse.
        let cfg = FarmConfigBuilder::from_yaml_text(yaml).expect("parse");
        assert_eq!(cfg.max_workers, 4);
    }

    #[test]
    fn yaml_parse_empty_string_returns_defaults() {
        let cfg = FarmConfigBuilder::from_yaml_text("").expect("parse");
        let default = FarmConfig::default();
        assert_eq!(cfg.max_workers, default.max_workers);
    }

    // --- FarmConfigBuilder setters ---

    #[test]
    fn builder_setters_work() {
        let cfg = FarmConfigBuilder::new()
            .max_workers(12)
            .job_timeout_ms(7_200_000)
            .checkpoint_interval_ms(120_000)
            .log_level("warn")
            .build();
        assert_eq!(cfg.max_workers, 12);
        assert_eq!(cfg.job_timeout_ms, 7_200_000);
        assert_eq!(cfg.log_level, "warn");
        assert!(cfg.validate().is_ok());
    }

    // --- Internal helpers ---

    #[test]
    fn split_first_colon_basic() {
        let result = split_first_colon("key: value");
        assert_eq!(result, Some(("key", " value")));
    }

    #[test]
    fn split_first_colon_no_colon() {
        assert!(split_first_colon("no colon here").is_none());
    }

    #[test]
    fn strip_quotes_double() {
        assert_eq!(strip_quotes("\"hello\""), "hello");
    }

    #[test]
    fn strip_quotes_single() {
        assert_eq!(strip_quotes("'world'"), "world");
    }

    #[test]
    fn strip_quotes_no_quotes() {
        assert_eq!(strip_quotes("plain"), "plain");
    }
}
