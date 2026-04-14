//! Metadata tampering detection.
//!
//! Compares an image's *embedded* metadata (read from the file itself, e.g. EXIF)
//! against an *external* reference (e.g. a database record or provenance log).
//! Discrepancies are collected and returned as human-readable strings.
//!
//! # Example
//!
//! ```
//! use oximedia_forensics::tamper_detect::{MetadataTamperDetector, MetadataMap};
//! use std::collections::HashMap;
//!
//! let mut embedded = HashMap::new();
//! embedded.insert("camera_model".to_string(), "CanonEOS".to_string());
//! embedded.insert("timestamp".to_string(), "2024-01-01T00:00:00Z".to_string());
//!
//! let mut external = HashMap::new();
//! external.insert("camera_model".to_string(), "CanonEOS".to_string());
//! external.insert("timestamp".to_string(), "2024-06-15T12:00:00Z".to_string());
//!
//! let issues = MetadataTamperDetector::check(&embedded, &external);
//! assert!(!issues.is_empty());
//! ```

#![allow(dead_code)]

use std::collections::HashMap;

/// A flat key-value metadata store.
///
/// Both embedded (in-file) and external (reference) metadata are represented
/// using this type.
pub type MetadataMap = HashMap<String, String>;

// ---------------------------------------------------------------------------
// MetadataTamperDetector
// ---------------------------------------------------------------------------

/// Compares embedded metadata against an external reference and returns a list
/// of discrepancies.
pub struct MetadataTamperDetector;

impl MetadataTamperDetector {
    /// Check `embedded` metadata against `external` reference metadata.
    ///
    /// Rules applied:
    ///
    /// 1. **Value mismatch** — a key present in both maps but with different
    ///    values is flagged.
    /// 2. **Missing key** — a key present in `external` but absent from
    ///    `embedded` is flagged (the file may have had its metadata stripped).
    /// 3. **Extra key** — a key present in `embedded` but absent from
    ///    `external` is reported as informational (unexpected metadata).
    ///
    /// # Returns
    ///
    /// A `Vec<String>` of human-readable discrepancy descriptions.  An empty
    /// vec means no tampering was detected.
    #[must_use]
    pub fn check(embedded: &MetadataMap, external: &MetadataMap) -> Vec<String> {
        let mut issues = Vec::new();

        // 1 & 2: iterate over external keys
        for (key, ext_value) in external {
            match embedded.get(key) {
                None => {
                    issues.push(format!(
                        "Missing key '{}': expected '{}' but key is absent from embedded metadata",
                        key, ext_value
                    ));
                }
                Some(emb_value) if emb_value != ext_value => {
                    issues.push(format!(
                        "Value mismatch for '{}': embedded='{}' external='{}'",
                        key, emb_value, ext_value
                    ));
                }
                Some(_) => { /* match — no issue */ }
            }
        }

        // 3: extra keys in embedded that are not in external
        for (key, emb_value) in embedded {
            if !external.contains_key(key) {
                issues.push(format!(
                    "Unexpected key '{}' in embedded metadata (value='{}') not present in reference",
                    key, emb_value
                ));
            }
        }

        // Sort for deterministic output
        issues.sort();
        issues
    }

    /// Strict check: returns `true` only when embedded and external are
    /// identical (same keys, same values).
    #[must_use]
    pub fn is_identical(embedded: &MetadataMap, external: &MetadataMap) -> bool {
        Self::check(embedded, external).is_empty()
    }

    /// Returns only the value-mismatch issues (ignores missing / extra keys).
    #[must_use]
    pub fn mismatches(embedded: &MetadataMap, external: &MetadataMap) -> Vec<String> {
        Self::check(embedded, external)
            .into_iter()
            .filter(|s| s.starts_with("Value mismatch"))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn map(pairs: &[(&str, &str)]) -> MetadataMap {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    // ── check ────────────────────────────────────────────────────────────────

    #[test]
    fn test_check_identical_maps_returns_empty() {
        let m = map(&[("camera", "Canon"), ("iso", "400")]);
        let issues = MetadataTamperDetector::check(&m, &m);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_check_value_mismatch_detected() {
        let embedded = map(&[("timestamp", "2024-01-01")]);
        let external = map(&[("timestamp", "2024-06-15")]);
        let issues = MetadataTamperDetector::check(&embedded, &external);
        assert_eq!(issues.len(), 1);
        assert!(issues[0].contains("Value mismatch"));
        assert!(issues[0].contains("timestamp"));
    }

    #[test]
    fn test_check_missing_key_detected() {
        let embedded = map(&[]); // no keys
        let external = map(&[("camera_model", "Nikon")]);
        let issues = MetadataTamperDetector::check(&embedded, &external);
        assert_eq!(issues.len(), 1);
        assert!(issues[0].contains("Missing key"));
        assert!(issues[0].contains("camera_model"));
    }

    #[test]
    fn test_check_extra_key_reported() {
        let embedded = map(&[("extra_key", "unexpected_value")]);
        let external = map(&[]);
        let issues = MetadataTamperDetector::check(&embedded, &external);
        assert_eq!(issues.len(), 1);
        assert!(issues[0].contains("Unexpected key"));
    }

    #[test]
    fn test_check_multiple_issues() {
        let embedded = map(&[("a", "1"), ("b", "WRONG"), ("extra", "x")]);
        let external = map(&[("a", "1"), ("b", "correct"), ("missing", "y")]);
        let issues = MetadataTamperDetector::check(&embedded, &external);
        // Value mismatch for 'b', missing 'missing', extra 'extra'
        assert_eq!(issues.len(), 3);
    }

    #[test]
    fn test_check_empty_both_maps() {
        let issues = MetadataTamperDetector::check(&map(&[]), &map(&[]));
        assert!(issues.is_empty());
    }

    #[test]
    fn test_check_returns_sorted_output() {
        let embedded = map(&[("z_key", "v1"), ("a_key", "BAD")]);
        let external = map(&[("z_key", "v1"), ("a_key", "GOOD")]);
        let issues = MetadataTamperDetector::check(&embedded, &external);
        assert_eq!(issues.len(), 1);
    }

    // ── is_identical ─────────────────────────────────────────────────────────

    #[test]
    fn test_is_identical_same_maps() {
        let m = map(&[("k", "v")]);
        assert!(MetadataTamperDetector::is_identical(&m, &m));
    }

    #[test]
    fn test_is_identical_different_maps() {
        let a = map(&[("k", "v1")]);
        let b = map(&[("k", "v2")]);
        assert!(!MetadataTamperDetector::is_identical(&a, &b));
    }

    // ── mismatches ────────────────────────────────────────────────────────────

    #[test]
    fn test_mismatches_only_value_mismatches() {
        let embedded = map(&[("a", "wrong"), ("extra", "x")]);
        let external = map(&[("a", "right"), ("missing", "y")]);
        let mm = MetadataTamperDetector::mismatches(&embedded, &external);
        // Only the value mismatch for 'a' should be included
        assert_eq!(mm.len(), 1);
        assert!(mm[0].contains("Value mismatch"));
    }
}
