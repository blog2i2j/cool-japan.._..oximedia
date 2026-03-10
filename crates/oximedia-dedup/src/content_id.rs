//! Content ID and fingerprinting for media assets.
//!
//! This module provides:
//! - `ContentId`: a UUID-like content identifier derived from FNV-128 hashing
//! - `ContentFingerprint`: combined audio/visual/metadata fingerprint
//! - `ContentIdRegistry`: registry for deduplication and lookup
//! - `ContentIdStats`: deduplication statistics

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// ContentId
// ---------------------------------------------------------------------------

/// A UUID-like content identifier derived from the data's FNV-128 hash.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ContentId(pub String);

impl ContentId {
    /// Generate a `ContentId` from arbitrary byte data using FNV-128.
    ///
    /// The result is formatted as a 32-character lowercase hex string.
    #[must_use]
    pub fn generate(data: &[u8]) -> Self {
        let hash = fnv128(data);
        // Format as two 64-bit hex segments
        let s = format!("{:016x}{:016x}", hash.0, hash.1);
        Self(s)
    }

    /// Return a string slice of the identifier.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ContentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// FNV-128 hash. Returns (high 64 bits, low 64 bits).
fn fnv128(data: &[u8]) -> (u64, u64) {
    // FNV-1a 128-bit: prime = 0x0000_0000_0000_0000_0000_0001_0000_0000_0000_0000_0000_013B
    // offset = 0x6c62272e07bb0142_62b821756295c58d
    let mut h_lo: u64 = 0x62b8_2175_6295_c58d;
    let mut h_hi: u64 = 0x6c62_272e_07bb_0142;

    for &byte in data {
        // XOR with byte (applied to low 64 bits only, as per FNV-128 spec approximation)
        h_lo ^= u64::from(byte);

        // Multiply by FNV prime (128-bit): 2^88 + 2^8 + 0x3b
        // We use 64-bit multiplication with split carry for the two 64-bit halves.
        // Prime low: 0x0000_0000_0000_013B, Prime high: 0x0000_0001_0000_0000
        let prime_lo: u64 = 0x0000_0000_0000_013b;
        let prime_hi: u64 = 0x0000_0001_0000_0000;

        let new_lo = h_lo.wrapping_mul(prime_lo);
        let carry = h_lo
            .wrapping_mul(prime_hi)
            .wrapping_add(h_hi.wrapping_mul(prime_lo));

        h_lo = new_lo;
        h_hi = carry;
    }

    (h_hi, h_lo)
}

// ---------------------------------------------------------------------------
// ContentFingerprint
// ---------------------------------------------------------------------------

/// A combined fingerprint for a media asset.
#[derive(Debug, Clone)]
pub struct ContentFingerprint {
    /// Content identifier.
    pub id: ContentId,
    /// Audio fingerprint codes (optional).
    pub audio_fingerprint: Option<Vec<u32>>,
    /// Visual fingerprint hashes (optional).
    pub visual_fingerprint: Option<Vec<u64>>,
    /// Metadata hash.
    pub metadata_hash: u64,
}

impl ContentFingerprint {
    /// Create a new fingerprint.
    #[must_use]
    pub fn new(
        id: ContentId,
        audio_fingerprint: Option<Vec<u32>>,
        visual_fingerprint: Option<Vec<u64>>,
        metadata_hash: u64,
    ) -> Self {
        Self {
            id,
            audio_fingerprint,
            visual_fingerprint,
            metadata_hash,
        }
    }
}

// ---------------------------------------------------------------------------
// ContentIdRegistry
// ---------------------------------------------------------------------------

/// Registry for content fingerprints supporting deduplication and lookup.
pub struct ContentIdRegistry {
    /// Stored fingerprints indexed by `ContentId`.
    fingerprints: Vec<ContentFingerprint>,
    /// Deduplication statistics.
    stats: ContentIdStats,
}

impl ContentIdRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            fingerprints: Vec::new(),
            stats: ContentIdStats::default(),
        }
    }

    /// Register a fingerprint.
    ///
    /// If an exact `ContentId` already exists, the duplicate count is incremented.
    pub fn register(&mut self, fingerprint: ContentFingerprint) {
        let is_duplicate = self.fingerprints.iter().any(|fp| fp.id == fingerprint.id);

        if is_duplicate {
            self.stats.duplicates_found += 1;
        } else {
            self.stats.total_registered += 1;
            self.fingerprints.push(fingerprint);
        }
    }

    /// Look up a fingerprint by `ContentId`.
    #[must_use]
    pub fn lookup(&self, id: &ContentId) -> Option<&ContentFingerprint> {
        self.fingerprints.iter().find(|fp| &fp.id == id)
    }

    /// Find fingerprints whose audio fingerprint matches the query.
    ///
    /// `min_match` is the minimum fraction of matching 32-bit codes (0.0–1.0).
    #[must_use]
    pub fn find_by_audio(&self, query: &[u32], min_match: f32) -> Vec<(ContentId, f32)> {
        self.fingerprints
            .iter()
            .filter_map(|fp| {
                fp.audio_fingerprint.as_ref().map(|audio| {
                    let sim = audio_code_similarity(query, audio);
                    (fp.id.clone(), sim)
                })
            })
            .filter(|(_, sim)| *sim >= min_match)
            .collect()
    }

    /// Find fingerprints whose visual fingerprint is similar to the query.
    ///
    /// `min_match` is the minimum Jaccard similarity.
    #[must_use]
    pub fn find_by_visual(&self, query: &[u64], min_match: f32) -> Vec<(ContentId, f32)> {
        self.fingerprints
            .iter()
            .filter_map(|fp| {
                fp.visual_fingerprint.as_ref().map(|visual| {
                    let sim = visual_hash_similarity(query, visual);
                    (fp.id.clone(), sim)
                })
            })
            .filter(|(_, sim)| *sim >= min_match)
            .collect()
    }

    /// Get current statistics.
    #[must_use]
    pub fn stats(&self) -> &ContentIdStats {
        &self.stats
    }

    /// Return the number of registered (unique) fingerprints.
    #[must_use]
    pub fn len(&self) -> usize {
        self.fingerprints.len()
    }

    /// Return true if no fingerprints have been registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fingerprints.is_empty()
    }
}

impl Default for ContentIdRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute fraction of matching 32-bit audio codes.
///
/// `matching = count of codes in query that appear in candidate` / max(len_a, len_b).
fn audio_code_similarity(query: &[u32], candidate: &[u32]) -> f32 {
    let denom = query.len().max(candidate.len());
    if denom == 0 {
        return 1.0;
    }

    let matches = query.iter().filter(|code| candidate.contains(code)).count();

    matches as f32 / denom as f32
}

/// Compute Jaccard similarity between two sets of visual hashes.
fn visual_hash_similarity(query: &[u64], candidate: &[u64]) -> f32 {
    let intersection = query.iter().filter(|h| candidate.contains(h)).count();
    let union = query.len() + candidate.len() - intersection;
    if union == 0 {
        return 1.0;
    }
    intersection as f32 / union as f32
}

// ---------------------------------------------------------------------------
// ContentIdStats
// ---------------------------------------------------------------------------

/// Statistics tracked by `ContentIdRegistry`.
#[derive(Debug, Clone, Default)]
pub struct ContentIdStats {
    /// Total number of unique items registered.
    pub total_registered: u64,
    /// Number of duplicates detected (not re-stored).
    pub duplicates_found: u64,
    /// Estimated storage saved in bytes (placeholder).
    pub storage_saved_bytes: u64,
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- ContentId tests ---

    #[test]
    fn test_content_id_generate_length() {
        let id = ContentId::generate(b"Hello, World!");
        assert_eq!(id.0.len(), 32);
    }

    #[test]
    fn test_content_id_generate_hex_chars() {
        let id = ContentId::generate(b"test data");
        assert!(id.0.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_content_id_deterministic() {
        let id1 = ContentId::generate(b"same input");
        let id2 = ContentId::generate(b"same input");
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_content_id_different_inputs() {
        let id1 = ContentId::generate(b"input A");
        let id2 = ContentId::generate(b"input B");
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_content_id_empty_data() {
        let id = ContentId::generate(b"");
        assert_eq!(id.0.len(), 32);
    }

    #[test]
    fn test_content_id_display() {
        let id = ContentId::generate(b"display test");
        let s = format!("{id}");
        assert_eq!(s.len(), 32);
    }

    // --- ContentFingerprint / Registry tests ---

    fn make_fp(data: &[u8]) -> ContentFingerprint {
        ContentFingerprint::new(
            ContentId::generate(data),
            Some(vec![1u32, 2, 3, 4]),
            Some(vec![10u64, 20, 30]),
            0xDEAD_BEEF,
        )
    }

    #[test]
    fn test_registry_empty() {
        let registry = ContentIdRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_registry_register_and_lookup() {
        let mut registry = ContentIdRegistry::new();
        let fp = make_fp(b"video1.mp4");
        let id = fp.id.clone();
        registry.register(fp);

        let found = registry.lookup(&id);
        assert!(found.is_some());
        assert_eq!(found.expect("operation should succeed").id, id);
    }

    #[test]
    fn test_registry_duplicate_not_stored() {
        let mut registry = ContentIdRegistry::new();
        let fp1 = make_fp(b"video1.mp4");
        let fp2 = make_fp(b"video1.mp4"); // same ID
        registry.register(fp1);
        registry.register(fp2);

        assert_eq!(registry.len(), 1);
        assert_eq!(registry.stats().duplicates_found, 1);
    }

    #[test]
    fn test_registry_find_by_audio_match() {
        let mut registry = ContentIdRegistry::new();
        let fp = ContentFingerprint::new(
            ContentId::generate(b"audio test"),
            Some(vec![1u32, 2, 3, 4, 5]),
            None,
            0,
        );
        registry.register(fp);

        // Query with same codes → high similarity
        let results = registry.find_by_audio(&[1, 2, 3, 4, 5], 0.9);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, 1.0);
    }

    #[test]
    fn test_registry_find_by_audio_no_match() {
        let mut registry = ContentIdRegistry::new();
        let fp = ContentFingerprint::new(
            ContentId::generate(b"audio test"),
            Some(vec![100u32, 200, 300]),
            None,
            0,
        );
        registry.register(fp);

        // Completely different codes
        let results = registry.find_by_audio(&[1, 2, 3], 0.5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_registry_find_by_visual_match() {
        let mut registry = ContentIdRegistry::new();
        let fp = ContentFingerprint::new(
            ContentId::generate(b"visual test"),
            None,
            Some(vec![10u64, 20, 30, 40]),
            0,
        );
        registry.register(fp);

        let results = registry.find_by_visual(&[10, 20, 30, 40], 0.9);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, 1.0);
    }

    #[test]
    fn test_registry_stats_initial() {
        let registry = ContentIdRegistry::new();
        assert_eq!(registry.stats().total_registered, 0);
        assert_eq!(registry.stats().duplicates_found, 0);
    }

    #[test]
    fn test_registry_multiple_unique() {
        let mut registry = ContentIdRegistry::new();
        for i in 0u8..5 {
            registry.register(make_fp(&[i]));
        }
        assert_eq!(registry.len(), 5);
        assert_eq!(registry.stats().total_registered, 5);
        assert_eq!(registry.stats().duplicates_found, 0);
    }

    #[test]
    fn test_audio_code_similarity_empty() {
        let sim = audio_code_similarity(&[], &[]);
        assert_eq!(sim, 1.0);
    }

    #[test]
    fn test_audio_code_similarity_disjoint() {
        let sim = audio_code_similarity(&[1, 2, 3], &[4, 5, 6]);
        assert_eq!(sim, 0.0);
    }
}
