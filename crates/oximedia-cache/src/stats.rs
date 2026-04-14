//! Cache statistics: hit-rate tracking and JSON export.
//!
//! [`CacheStats`] records hits and misses in an atomic-friendly manner and
//! derives the hit rate on demand.  A JSON snapshot can be exported via
//! `to_json()` without any external serialisation dependency.
//!
//! # Example
//!
//! ```
//! use oximedia_cache::stats::CacheStats;
//!
//! let mut stats = CacheStats::new();
//! stats.record_hit();
//! stats.record_hit();
//! stats.record_miss();
//! assert!((stats.hit_rate() - 2.0 / 3.0).abs() < 1e-6);
//! let json = stats.to_json();
//! assert!(json.contains("\"hits\":2"));
//! ```

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// CacheStats
// ---------------------------------------------------------------------------

/// Accumulates cache hit/miss counters and produces derived statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl CacheStats {
    /// Create a new `CacheStats` with all counters at zero.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one cache hit.
    pub fn record_hit(&mut self) {
        self.hits = self.hits.saturating_add(1);
    }

    /// Record one cache miss.
    pub fn record_miss(&mut self) {
        self.misses = self.misses.saturating_add(1);
    }

    /// Record one eviction.
    pub fn record_eviction(&mut self) {
        self.evictions = self.evictions.saturating_add(1);
    }

    /// Total number of hits.
    #[must_use]
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Total number of misses.
    #[must_use]
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Total number of evictions.
    #[must_use]
    pub fn evictions(&self) -> u64 {
        self.evictions
    }

    /// Total number of lookups (hits + misses).
    #[must_use]
    pub fn total_lookups(&self) -> u64 {
        self.hits.saturating_add(self.misses)
    }

    /// Hit rate as a fraction in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` when no lookups have occurred.
    #[must_use]
    pub fn hit_rate(&self) -> f32 {
        let total = self.total_lookups();
        if total == 0 {
            return 0.0;
        }
        self.hits as f32 / total as f32
    }

    /// Miss rate as a fraction in `[0.0, 1.0]`.
    #[must_use]
    pub fn miss_rate(&self) -> f32 {
        1.0 - self.hit_rate()
    }

    /// Export statistics as a compact JSON string.
    ///
    /// # Format
    ///
    /// ```json
    /// {"hits":10,"misses":5,"evictions":2,"hit_rate":0.666667}
    /// ```
    #[must_use]
    pub fn to_json(&self) -> String {
        format!(
            "{{\"hits\":{},\"misses\":{},\"evictions\":{},\"hit_rate\":{:.6}}}",
            self.hits,
            self.misses,
            self.evictions,
            self.hit_rate()
        )
    }

    /// Reset all counters to zero.
    pub fn reset(&mut self) {
        self.hits = 0;
        self.misses = 0;
        self.evictions = 0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── new ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_new_starts_at_zero() {
        let s = CacheStats::new();
        assert_eq!(s.hits(), 0);
        assert_eq!(s.misses(), 0);
        assert_eq!(s.evictions(), 0);
    }

    // ── record_hit / record_miss ──────────────────────────────────────────────

    #[test]
    fn test_record_hit_increments() {
        let mut s = CacheStats::new();
        s.record_hit();
        s.record_hit();
        assert_eq!(s.hits(), 2);
    }

    #[test]
    fn test_record_miss_increments() {
        let mut s = CacheStats::new();
        s.record_miss();
        assert_eq!(s.misses(), 1);
    }

    #[test]
    fn test_record_eviction_increments() {
        let mut s = CacheStats::new();
        s.record_eviction();
        s.record_eviction();
        assert_eq!(s.evictions(), 2);
    }

    // ── hit_rate ─────────────────────────────────────────────────────────────

    #[test]
    fn test_hit_rate_zero_when_no_lookups() {
        let s = CacheStats::new();
        assert!((s.hit_rate() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_hit_rate_one_hundred_percent() {
        let mut s = CacheStats::new();
        s.record_hit();
        assert!((s.hit_rate() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hit_rate_two_thirds() {
        let mut s = CacheStats::new();
        s.record_hit();
        s.record_hit();
        s.record_miss();
        assert!((s.hit_rate() - 2.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_miss_rate_complement() {
        let mut s = CacheStats::new();
        s.record_hit();
        s.record_miss();
        let hit = s.hit_rate();
        let miss = s.miss_rate();
        assert!((hit + miss - 1.0).abs() < 1e-6);
    }

    // ── total_lookups ─────────────────────────────────────────────────────────

    #[test]
    fn test_total_lookups_sums_hits_and_misses() {
        let mut s = CacheStats::new();
        s.record_hit();
        s.record_hit();
        s.record_miss();
        assert_eq!(s.total_lookups(), 3);
    }

    // ── to_json ───────────────────────────────────────────────────────────────

    #[test]
    fn test_to_json_contains_hits() {
        let mut s = CacheStats::new();
        s.record_hit();
        s.record_hit();
        let json = s.to_json();
        assert!(json.contains("\"hits\":2"), "JSON: {json}");
    }

    #[test]
    fn test_to_json_contains_misses() {
        let mut s = CacheStats::new();
        s.record_miss();
        let json = s.to_json();
        assert!(json.contains("\"misses\":1"), "JSON: {json}");
    }

    #[test]
    fn test_to_json_contains_evictions() {
        let mut s = CacheStats::new();
        s.record_eviction();
        let json = s.to_json();
        assert!(json.contains("\"evictions\":1"), "JSON: {json}");
    }

    #[test]
    fn test_to_json_contains_hit_rate() {
        let mut s = CacheStats::new();
        s.record_hit();
        s.record_miss();
        let json = s.to_json();
        assert!(json.contains("\"hit_rate\":"), "JSON: {json}");
    }

    #[test]
    fn test_to_json_is_valid_braces() {
        let s = CacheStats::new();
        let json = s.to_json();
        assert!(json.starts_with('{'));
        assert!(json.ends_with('}'));
    }

    // ── reset ────────────────────────────────────────────────────────────────

    #[test]
    fn test_reset_clears_all_counters() {
        let mut s = CacheStats::new();
        s.record_hit();
        s.record_miss();
        s.record_eviction();
        s.reset();
        assert_eq!(s.hits(), 0);
        assert_eq!(s.misses(), 0);
        assert_eq!(s.evictions(), 0);
    }
}
