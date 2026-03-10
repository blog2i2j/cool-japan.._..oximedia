//! Frame-level descriptor matching for video alignment in `OxiMedia`.
//!
//! Uses perceptual hashing (binary descriptors) and Hamming distance to
//! find the best corresponding frame between two video streams.

#![allow(dead_code)]

/// A binary feature descriptor stored as a fixed-width bit array.
///
/// Uses 256 bits (32 bytes) – compatible with ORB / BRIEF descriptors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameDescriptor {
    /// Raw binary descriptor bytes.
    pub data: [u8; 32],
    /// Frame presentation timestamp in milliseconds.
    pub pts_ms: i64,
}

impl FrameDescriptor {
    /// Create a descriptor from raw bytes and a PTS.
    #[must_use]
    pub fn new(data: [u8; 32], pts_ms: i64) -> Self {
        Self { data, pts_ms }
    }

    /// Compute the Hamming distance to another descriptor.
    ///
    /// Returns a value in `[0, 256]`; lower is more similar.
    #[must_use]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }

    /// Return `true` if descriptors are identical.
    #[must_use]
    pub fn is_identical(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

/// A candidate match between a query frame and a database frame.
#[derive(Debug, Clone)]
pub struct MatchCandidate {
    /// Hamming distance between the two descriptors.
    pub hamming: u32,
    /// PTS of the matched frame in the reference stream (ms).
    pub ref_pts_ms: i64,
    /// PTS of the query frame (ms).
    pub query_pts_ms: i64,
    /// Normalised confidence `[0.0, 1.0]`; derived from Hamming distance.
    pub confidence: f64,
}

impl MatchCandidate {
    /// Create a match candidate.
    ///
    /// `max_bits` is the descriptor width in bits used to normalise confidence.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn new(hamming: u32, ref_pts_ms: i64, query_pts_ms: i64, max_bits: u32) -> Self {
        let confidence = if max_bits == 0 {
            0.0
        } else {
            1.0 - (f64::from(hamming) / f64::from(max_bits))
        };
        Self {
            hamming,
            ref_pts_ms,
            query_pts_ms,
            confidence,
        }
    }

    /// Return `true` when confidence meets the given threshold.
    #[must_use]
    pub fn confidence_ok(&self, threshold: f64) -> bool {
        self.confidence >= threshold
    }

    /// Signed temporal offset: reference PTS minus query PTS, in ms.
    #[must_use]
    pub fn offset_ms(&self) -> i64 {
        self.ref_pts_ms - self.query_pts_ms
    }
}

/// Configuration for [`FrameMatcher`].
#[derive(Debug, Clone)]
pub struct FrameMatcherConfig {
    /// Maximum Hamming distance to consider a valid match.
    pub max_hamming: u32,
    /// Minimum confidence threshold for [`MatchCandidate::confidence_ok`].
    pub min_confidence: f64,
    /// Descriptor bit width (default 256 for 32-byte descriptors).
    pub descriptor_bits: u32,
}

impl Default for FrameMatcherConfig {
    fn default() -> Self {
        Self {
            max_hamming: 64,
            min_confidence: 0.75,
            descriptor_bits: 256,
        }
    }
}

/// Matches a query frame descriptor against a reference database.
#[derive(Debug)]
pub struct FrameMatcher {
    config: FrameMatcherConfig,
    reference: Vec<FrameDescriptor>,
}

impl FrameMatcher {
    /// Create a new matcher with the given configuration.
    #[must_use]
    pub fn new(config: FrameMatcherConfig) -> Self {
        Self {
            config,
            reference: Vec::new(),
        }
    }

    /// Create a matcher with default configuration.
    #[must_use]
    pub fn default_matcher() -> Self {
        Self::new(FrameMatcherConfig::default())
    }

    /// Add a reference frame descriptor.
    pub fn add_reference(&mut self, desc: FrameDescriptor) {
        self.reference.push(desc);
    }

    /// Load a collection of reference descriptors.
    pub fn load_reference(&mut self, descs: Vec<FrameDescriptor>) {
        self.reference = descs;
    }

    /// Find all candidates within the configured Hamming threshold.
    #[must_use]
    pub fn find_match(&self, query: &FrameDescriptor) -> Vec<MatchCandidate> {
        let bits = self.config.descriptor_bits;
        let max_h = self.config.max_hamming;
        self.reference
            .iter()
            .filter_map(|r| {
                let h = query.hamming_distance(r);
                if h <= max_h {
                    Some(MatchCandidate::new(h, r.pts_ms, query.pts_ms, bits))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Return the single best (lowest Hamming distance) match, if any.
    #[must_use]
    pub fn best_match(&self, query: &FrameDescriptor) -> Option<MatchCandidate> {
        let bits = self.config.descriptor_bits;
        let max_h = self.config.max_hamming;
        self.reference
            .iter()
            .filter_map(|r| {
                let h = query.hamming_distance(r);
                if h <= max_h {
                    Some(MatchCandidate::new(h, r.pts_ms, query.pts_ms, bits))
                } else {
                    None
                }
            })
            .min_by_key(|c| c.hamming)
    }

    /// Number of loaded reference descriptors.
    #[must_use]
    pub fn reference_count(&self) -> usize {
        self.reference.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zeros() -> [u8; 32] {
        [0u8; 32]
    }
    fn ones() -> [u8; 32] {
        [0xFFu8; 32]
    }
    fn half() -> [u8; 32] {
        let mut d = [0u8; 32];
        for i in 0..16 {
            d[i] = 0xFF;
        }
        d
    }

    // ── FrameDescriptor ──────────────────────────────────────────────────────

    #[test]
    fn test_hamming_identical() {
        let d = FrameDescriptor::new(zeros(), 0);
        assert_eq!(d.hamming_distance(&d), 0);
    }

    #[test]
    fn test_hamming_all_different() {
        let a = FrameDescriptor::new(zeros(), 0);
        let b = FrameDescriptor::new(ones(), 0);
        assert_eq!(a.hamming_distance(&b), 256);
    }

    #[test]
    fn test_hamming_half() {
        let a = FrameDescriptor::new(zeros(), 0);
        let b = FrameDescriptor::new(half(), 0);
        assert_eq!(a.hamming_distance(&b), 128);
    }

    #[test]
    fn test_is_identical_true() {
        let a = FrameDescriptor::new(zeros(), 100);
        let b = FrameDescriptor::new(zeros(), 200);
        assert!(a.is_identical(&b));
    }

    #[test]
    fn test_is_identical_false() {
        let a = FrameDescriptor::new(zeros(), 0);
        let b = FrameDescriptor::new(ones(), 0);
        assert!(!a.is_identical(&b));
    }

    // ── MatchCandidate ───────────────────────────────────────────────────────

    #[test]
    fn test_candidate_perfect_confidence() {
        let c = MatchCandidate::new(0, 1000, 1000, 256);
        assert!((c.confidence - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_candidate_zero_confidence() {
        let c = MatchCandidate::new(256, 0, 0, 256);
        assert!(c.confidence < f64::EPSILON);
    }

    #[test]
    fn test_candidate_confidence_ok() {
        let c = MatchCandidate::new(32, 0, 0, 256);
        assert!(c.confidence_ok(0.75));
        assert!(!c.confidence_ok(0.999));
    }

    #[test]
    fn test_candidate_offset_ms() {
        let c = MatchCandidate::new(10, 2000, 1500, 256);
        assert_eq!(c.offset_ms(), 500);
    }

    #[test]
    fn test_candidate_zero_max_bits() {
        let c = MatchCandidate::new(10, 0, 0, 0);
        assert!((c.confidence).abs() < f64::EPSILON);
    }

    // ── FrameMatcher ─────────────────────────────────────────────────────────

    #[test]
    fn test_matcher_empty_reference() {
        let matcher = FrameMatcher::default_matcher();
        let query = FrameDescriptor::new(zeros(), 0);
        assert!(matcher.find_match(&query).is_empty());
        assert!(matcher.best_match(&query).is_none());
    }

    #[test]
    fn test_matcher_finds_exact() {
        let mut matcher = FrameMatcher::default_matcher();
        matcher.add_reference(FrameDescriptor::new(zeros(), 1000));
        let query = FrameDescriptor::new(zeros(), 500);
        let candidates = matcher.find_match(&query);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].hamming, 0);
    }

    #[test]
    fn test_matcher_best_match_closest() {
        let mut matcher = FrameMatcher::default_matcher();
        let mut near = zeros();
        near[0] = 0x01; // Hamming = 1
        let mut far = zeros();
        far[0] = 0x0F; // Hamming = 4
        matcher.add_reference(FrameDescriptor::new(near, 1000));
        matcher.add_reference(FrameDescriptor::new(far, 2000));
        let query = FrameDescriptor::new(zeros(), 0);
        let best = matcher.best_match(&query).expect("best should be valid");
        assert_eq!(best.hamming, 1);
        assert_eq!(best.ref_pts_ms, 1000);
    }

    #[test]
    fn test_matcher_rejects_beyond_threshold() {
        let cfg = FrameMatcherConfig {
            max_hamming: 10,
            ..Default::default()
        };
        let mut matcher = FrameMatcher::new(cfg);
        matcher.add_reference(FrameDescriptor::new(ones(), 500));
        let query = FrameDescriptor::new(zeros(), 0);
        // Hamming = 256 > 10 → no match
        assert!(matcher.find_match(&query).is_empty());
    }

    #[test]
    fn test_matcher_reference_count() {
        let mut matcher = FrameMatcher::default_matcher();
        assert_eq!(matcher.reference_count(), 0);
        matcher.add_reference(FrameDescriptor::new(zeros(), 0));
        matcher.add_reference(FrameDescriptor::new(ones(), 1));
        assert_eq!(matcher.reference_count(), 2);
    }

    #[test]
    fn test_load_reference_replaces() {
        let mut matcher = FrameMatcher::default_matcher();
        matcher.add_reference(FrameDescriptor::new(zeros(), 0));
        let new_refs = vec![
            FrameDescriptor::new(ones(), 100),
            FrameDescriptor::new(half(), 200),
        ];
        matcher.load_reference(new_refs);
        assert_eq!(matcher.reference_count(), 2);
    }
}
