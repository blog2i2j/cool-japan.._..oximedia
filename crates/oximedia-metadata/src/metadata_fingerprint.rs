//! Stable hash-based fingerprinting for metadata records.
//!
//! Computes a deterministic, order-independent fingerprint over a set of metadata
//! key-value pairs. The fingerprint can be used for:
//!
//! - Cache invalidation (has this record changed?)
//! - Deduplication (are these two records semantically identical?)
//! - Change detection across format conversions
//!
//! # Algorithm
//!
//! Each key-value pair is hashed independently using a FNV-1a 64-bit hash
//! (chosen for its speed and lack of patent/license issues). Individual pair
//! hashes are combined via XOR so the final result is order-independent.
//! A final mixing step applies a non-linear bijection (xorshift64*) to improve
//! avalanche behaviour before the 64-bit result is formatted as a 16-character
//! hex string.
//!
//! # Example
//!
//! ```
//! use oximedia_metadata::metadata_fingerprint::MetadataFingerprint;
//! use std::collections::HashMap;
//!
//! let mut fields: HashMap<String, String> = HashMap::new();
//! fields.insert("title".to_string(), "My Song".to_string());
//! fields.insert("artist".to_string(), "Artist".to_string());
//!
//! let fp = MetadataFingerprint::from_fields(&fields);
//! println!("fingerprint: {}", fp.hex());
//! ```

use std::collections::HashMap;

// ─── FNV-1a 64-bit constants ────────────────────────────────────────────────

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Compute FNV-1a 64-bit hash over a byte slice.
fn fnv1a_64(data: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Compute FNV-1a 64-bit hash over a string.
fn fnv1a_str(s: &str) -> u64 {
    fnv1a_64(s.as_bytes())
}

/// xorshift64* mixing step to improve avalanche.
fn mix64(mut x: u64) -> u64 {
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    x.wrapping_mul(0x2545_f491_4f6c_dd1d)
}

// ─── Public types ────────────────────────────────────────────────────────────

/// A stable, order-independent fingerprint derived from metadata key-value pairs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MetadataFingerprint {
    /// The raw 64-bit fingerprint value.
    pub raw: u64,
}

impl MetadataFingerprint {
    /// Create a fingerprint from an iterator of `(key, value)` string pairs.
    ///
    /// The order of iteration does not affect the result.
    pub fn from_pairs<'a, I>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (&'a str, &'a str)>,
    {
        let mut combined: u64 = 0;
        for (key, value) in pairs {
            // Hash the pair as "key\x00value" to avoid key="a", value="bcd"
            // colliding with key="ab", value="cd".
            let pair_hash = fnv1a_64(
                &[key.as_bytes(), &[0u8], value.as_bytes()].concat(),
            );
            // XOR accumulation → order-independent
            combined ^= pair_hash;
        }
        // Final mixing
        Self { raw: mix64(combined) }
    }

    /// Create a fingerprint from a `HashMap<String, String>`.
    pub fn from_fields(fields: &HashMap<String, String>) -> Self {
        Self::from_pairs(fields.iter().map(|(k, v)| (k.as_str(), v.as_str())))
    }

    /// Return the fingerprint as a lowercase 16-character hexadecimal string.
    pub fn hex(&self) -> String {
        format!("{:016x}", self.raw)
    }

    /// Check whether two fingerprints match.
    pub fn matches(&self, other: &Self) -> bool {
        self.raw == other.raw
    }

    /// Combine two fingerprints using XOR + mixing (for hierarchical fingerprinting).
    pub fn combine(&self, other: &Self) -> Self {
        Self { raw: mix64(self.raw ^ other.raw) }
    }
}

impl std::fmt::Display for MetadataFingerprint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.hex())
    }
}

// ─── Incremental builder ─────────────────────────────────────────────────────

/// Builder for incrementally constructing a `MetadataFingerprint` without
/// materialising all fields at once.
#[derive(Debug, Default)]
pub struct FingerprintBuilder {
    accumulated: u64,
}

impl FingerprintBuilder {
    /// Create a new empty builder.
    pub fn new() -> Self {
        Self { accumulated: 0 }
    }

    /// Add a single `(key, value)` pair to the fingerprint.
    pub fn add(&mut self, key: &str, value: &str) -> &mut Self {
        let pair_hash = fnv1a_64(
            &[key.as_bytes(), &[0u8], value.as_bytes()].concat(),
        );
        self.accumulated ^= pair_hash;
        self
    }

    /// Finalise and return the `MetadataFingerprint`.
    pub fn finish(&self) -> MetadataFingerprint {
        MetadataFingerprint { raw: mix64(self.accumulated) }
    }
}

// ─── Diff helpers ────────────────────────────────────────────────────────────

/// Result of comparing two metadata fingerprints alongside their source fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FingerprintDiff {
    /// The fingerprints are identical; the fields have not changed.
    Identical,
    /// The fingerprints differ; fields were added, removed, or changed.
    Changed {
        /// Keys present in `before` but not in `after`.
        removed: Vec<String>,
        /// Keys present in `after` but not in `before`.
        added: Vec<String>,
        /// Keys present in both but with different values.
        modified: Vec<String>,
    },
}

/// Compare two field maps and produce a `FingerprintDiff`.
pub fn diff_fields(
    before: &HashMap<String, String>,
    after: &HashMap<String, String>,
) -> FingerprintDiff {
    let fp_before = MetadataFingerprint::from_fields(before);
    let fp_after = MetadataFingerprint::from_fields(after);

    if fp_before == fp_after {
        return FingerprintDiff::Identical;
    }

    let mut removed = Vec::new();
    let mut added = Vec::new();
    let mut modified = Vec::new();

    for key in before.keys() {
        if let Some(after_val) = after.get(key) {
            if before.get(key) != Some(after_val) {
                modified.push(key.clone());
            }
        } else {
            removed.push(key.clone());
        }
    }
    for key in after.keys() {
        if !before.contains_key(key) {
            added.push(key.clone());
        }
    }

    removed.sort();
    added.sort();
    modified.sort();

    FingerprintDiff::Changed { removed, added, modified }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn fields(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn test_fingerprint_deterministic() {
        let f = fields(&[("title", "Song"), ("artist", "Band")]);
        let fp1 = MetadataFingerprint::from_fields(&f);
        let fp2 = MetadataFingerprint::from_fields(&f);
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_fingerprint_order_independent() {
        // Same fields, different insertion order via different pair iterators
        let fp1 = MetadataFingerprint::from_pairs([("title", "Song"), ("artist", "Band")]);
        let fp2 = MetadataFingerprint::from_pairs([("artist", "Band"), ("title", "Song")]);
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_fingerprint_different_values_differ() {
        let fp1 = MetadataFingerprint::from_pairs([("title", "Song A")]);
        let fp2 = MetadataFingerprint::from_pairs([("title", "Song B")]);
        assert_ne!(fp1, fp2);
    }

    #[test]
    fn test_fingerprint_different_keys_differ() {
        let fp1 = MetadataFingerprint::from_pairs([("title", "X")]);
        let fp2 = MetadataFingerprint::from_pairs([("artist", "X")]);
        assert_ne!(fp1, fp2);
    }

    #[test]
    fn test_fingerprint_hex_length() {
        let fp = MetadataFingerprint::from_pairs([("k", "v")]);
        assert_eq!(fp.hex().len(), 16);
    }

    #[test]
    fn test_fingerprint_hex_lowercase() {
        let fp = MetadataFingerprint::from_pairs([("k", "v")]);
        assert_eq!(fp.hex(), fp.hex().to_lowercase());
    }

    #[test]
    fn test_fingerprint_matches() {
        let fp1 = MetadataFingerprint::from_pairs([("a", "b")]);
        let fp2 = MetadataFingerprint::from_pairs([("a", "b")]);
        assert!(fp1.matches(&fp2));
    }

    #[test]
    fn test_fingerprint_empty_fields() {
        let fp = MetadataFingerprint::from_pairs(std::iter::empty::<(&str, &str)>());
        // Should produce a stable (mixed zero) result, not crash
        assert_eq!(fp.hex().len(), 16);
    }

    #[test]
    fn test_builder_same_as_from_pairs() {
        let mut builder = FingerprintBuilder::new();
        builder.add("title", "My Song");
        builder.add("artist", "My Artist");
        let fp_builder = builder.finish();

        let fp_direct = MetadataFingerprint::from_pairs([
            ("artist", "My Artist"),
            ("title", "My Song"),
        ]);
        assert_eq!(fp_builder, fp_direct);
    }

    #[test]
    fn test_combine_order_independent() {
        let fp1 = MetadataFingerprint::from_pairs([("a", "1")]);
        let fp2 = MetadataFingerprint::from_pairs([("b", "2")]);
        // XOR is commutative, so combine should be symmetric in raw input
        // (though mixing makes it non-trivially equal — check stability)
        let combined_ab = fp1.combine(&fp2);
        let combined_ba = fp2.combine(&fp1);
        assert_eq!(combined_ab, combined_ba);
    }

    #[test]
    fn test_diff_identical() {
        let f = fields(&[("title", "T"), ("artist", "A")]);
        let diff = diff_fields(&f, &f);
        assert_eq!(diff, FingerprintDiff::Identical);
    }

    #[test]
    fn test_diff_added_field() {
        let before = fields(&[("title", "T")]);
        let after = fields(&[("title", "T"), ("artist", "A")]);
        match diff_fields(&before, &after) {
            FingerprintDiff::Changed { added, removed, modified } => {
                assert_eq!(added, vec!["artist"]);
                assert!(removed.is_empty());
                assert!(modified.is_empty());
            }
            _ => panic!("expected Changed"),
        }
    }

    #[test]
    fn test_diff_removed_field() {
        let before = fields(&[("title", "T"), ("artist", "A")]);
        let after = fields(&[("title", "T")]);
        match diff_fields(&before, &after) {
            FingerprintDiff::Changed { removed, added, modified } => {
                assert_eq!(removed, vec!["artist"]);
                assert!(added.is_empty());
                assert!(modified.is_empty());
            }
            _ => panic!("expected Changed"),
        }
    }

    #[test]
    fn test_diff_modified_field() {
        let before = fields(&[("title", "Old Title")]);
        let after = fields(&[("title", "New Title")]);
        match diff_fields(&before, &after) {
            FingerprintDiff::Changed { modified, added, removed } => {
                assert_eq!(modified, vec!["title"]);
                assert!(added.is_empty());
                assert!(removed.is_empty());
            }
            _ => panic!("expected Changed"),
        }
    }

    #[test]
    fn test_display_is_hex() {
        let fp = MetadataFingerprint::from_pairs([("key", "val")]);
        assert_eq!(fp.to_string(), fp.hex());
    }

    #[test]
    fn test_key_value_collision_avoidance() {
        // "ab" + "cd" vs "a" + "bcd" should produce different hashes
        let fp1 = MetadataFingerprint::from_pairs([("ab", "cd")]);
        let fp2 = MetadataFingerprint::from_pairs([("a", "bcd")]);
        assert_ne!(fp1, fp2);
    }
}
