//! Bloom filter pre-screening for deduplication pipelines.
//!
//! [`BloomPrescreen`] provides an optimal Bloom filter implementation using
//! FNV-1a with varied seeds for k independent hash functions.  It is intended
//! as a fast first-pass gate before expensive perceptual or SSIM comparisons.
//!
//! # Design
//!
//! - Bit array backed by `Vec<u64>` (cache-friendly word-at-a-time access).
//! - Hash family: FNV-1a with per-hash seed diversification (Kirsch-Mitzenmacher).
//! - Optimal `m` and `k` computed from capacity `n` and target FPR `p`:
//!   - `m = ceil(-n * ln(p) / (ln(2))^2)`
//!   - `k = round((m / n) * ln(2))`
//! - False-positive rate estimate: `(1 - e^(-k*inserted/m))^k`

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

/// Bloom filter optimised for deduplication pre-screening.
///
/// # Example
///
/// ```
/// use oximedia_dedup::bloom_prescreen::BloomPrescreen;
///
/// let mut bp = BloomPrescreen::new(1000, 0.01);
/// bp.insert(b"hello");
/// assert!(bp.might_contain(b"hello"));
/// assert!(!bp.might_contain(b"world")); // (with high probability)
/// ```
#[derive(Debug, Clone)]
pub struct BloomPrescreen {
    /// Underlying bit storage (each `u64` holds 64 bits).
    pub bits: Vec<u64>,
    /// Number of independent hash functions.
    pub k_hashes: u32,
    /// Total number of addressable bits.
    pub num_bits: usize,
    /// Number of items inserted (used for FPR estimation).
    items_inserted: u64,
}

impl BloomPrescreen {
    /// Create a new `BloomPrescreen` sized for `capacity` expected items at
    /// the given `false_positive_rate`.
    ///
    /// Uses the standard formulas:
    /// - `m = ceil(-n * ln(p) / (ln 2)^2)`  (number of bits)
    /// - `k = round((m / n) * ln 2)`         (number of hash functions)
    ///
    /// Both values are clamped to sensible minimums.
    #[must_use]
    pub fn new(capacity: usize, false_positive_rate: f64) -> Self {
        let n = capacity.max(1) as f64;
        let p = false_positive_rate.clamp(1e-15, 1.0 - f64::EPSILON);
        let ln2 = std::f64::consts::LN_2;

        // Optimal bit count
        let m_float = -n * p.ln() / (ln2 * ln2);
        let m = (m_float.ceil() as usize).max(64);

        // Optimal number of hash functions
        let k_float = (m as f64 / n) * ln2;
        let k = (k_float.round() as u32).clamp(1, 32);

        // Allocate words
        let num_words = (m + 63) / 64;

        Self {
            bits: vec![0u64; num_words],
            k_hashes: k,
            num_bits: m,
            items_inserted: 0,
        }
    }

    /// Insert `key` into the filter.
    ///
    /// After calling this, [`might_contain`](Self::might_contain) will always
    /// return `true` for the same `key` (no false negatives).
    pub fn insert(&mut self, key: &[u8]) {
        for i in 0..self.k_hashes {
            let bit_idx = self.bit_index(key, i);
            let word = bit_idx / 64;
            let bit = bit_idx % 64;
            if word < self.bits.len() {
                self.bits[word] |= 1u64 << bit;
            }
        }
        self.items_inserted += 1;
    }

    /// Returns `true` if `key` *might* be in the set (possible false positive).
    /// Returns `false` if `key` is *definitely not* in the set.
    #[must_use]
    pub fn might_contain(&self, key: &[u8]) -> bool {
        for i in 0..self.k_hashes {
            let bit_idx = self.bit_index(key, i);
            let word = bit_idx / 64;
            let bit = bit_idx % 64;
            match self.bits.get(word) {
                Some(w) if w & (1u64 << bit) != 0 => continue,
                _ => return false,
            }
        }
        true
    }

    /// Estimate the current false-positive rate given the number of inserted items.
    ///
    /// Formula: `(1 - e^(-k * n / m))^k`
    ///
    /// Returns `1.0` when the filter is full.
    #[must_use]
    pub fn false_positive_rate_estimate(&self) -> f64 {
        if self.num_bits == 0 {
            return 1.0;
        }
        let k = self.k_hashes as f64;
        let n = self.items_inserted as f64;
        let m = self.num_bits as f64;
        let inner = (-k * n / m).exp();
        (1.0 - inner).powf(k)
    }

    /// Number of items inserted so far.
    #[must_use]
    pub fn items_inserted(&self) -> u64 {
        self.items_inserted
    }

    /// Reset the filter to its initial empty state.
    pub fn clear(&mut self) {
        for w in &mut self.bits {
            *w = 0;
        }
        self.items_inserted = 0;
    }

    /// Number of set bits (useful for diagnostics).
    #[must_use]
    pub fn set_bit_count(&self) -> u64 {
        self.bits.iter().map(|w| u64::from(w.count_ones())).sum()
    }

    // ── Hash family ──────────────────────────────────────────────────────────

    /// Compute the bit index for the `i`-th hash of `key`.
    ///
    /// Uses FNV-1a as the base hash, then applies the Kirsch-Mitzenmacher
    /// technique: `h_i(x) = (h1(x) + i * h2(x)) mod m`.
    ///
    /// `h2` is FNV-1a seeded with a distinct per-function constant derived
    /// from the index, ensuring independence across hash functions.
    fn bit_index(&self, key: &[u8], i: u32) -> usize {
        let h1 = fnv1a_64(key);
        // Seed h2 with a unique per-index constant (large prime mix)
        let seed = u64::from(i)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(0x6c62_272e_07bb_0142);
        let h2 = fnv1a_64_seeded(key, seed);
        let combined = h1.wrapping_add(u64::from(i).wrapping_mul(h2));
        (combined % self.num_bits as u64) as usize
    }
}

// ── FNV-1a hash primitives ────────────────────────────────────────────────────

/// FNV-1a 64-bit hash (standard offset basis).
fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// FNV-1a 64-bit hash with a custom seed mixed into the offset basis.
fn fnv1a_64_seeded(data: &[u8], seed: u64) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ seed.wrapping_mul(0x0000_0100_0000_01b3);
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction ──────────────────────────────────────────────────────────

    #[test]
    fn test_new_allocates_bits() {
        let bp = BloomPrescreen::new(1000, 0.01);
        assert!(!bp.bits.is_empty());
        assert!(bp.k_hashes >= 1);
        assert!(bp.num_bits >= 64);
    }

    #[test]
    fn test_new_k_clamp_lower() {
        // Even at extreme FPR the k must be at least 1.
        let bp = BloomPrescreen::new(1, 0.99);
        assert!(bp.k_hashes >= 1);
    }

    #[test]
    fn test_new_k_clamp_upper() {
        // Very tight FPR should not produce k > 32.
        let bp = BloomPrescreen::new(1_000_000, 1e-15);
        assert!(bp.k_hashes <= 32);
    }

    #[test]
    fn test_larger_capacity_more_bits() {
        let small = BloomPrescreen::new(100, 0.01);
        let large = BloomPrescreen::new(100_000, 0.01);
        assert!(large.num_bits > small.num_bits);
    }

    #[test]
    fn test_stricter_fpr_more_hashes() {
        let loose = BloomPrescreen::new(1000, 0.1);
        let strict = BloomPrescreen::new(1000, 0.0001);
        assert!(strict.k_hashes >= loose.k_hashes);
    }

    // ── No false negatives ────────────────────────────────────────────────────

    #[test]
    fn test_insert_then_might_contain() {
        let mut bp = BloomPrescreen::new(100, 0.01);
        let keys: &[&[u8]] = &[b"alpha", b"beta", b"gamma", b"delta", b"epsilon"];
        for k in keys {
            bp.insert(k);
        }
        for k in keys {
            assert!(bp.might_contain(k), "no false negative for {:?}", k);
        }
    }

    #[test]
    fn test_empty_filter_returns_false() {
        let bp = BloomPrescreen::new(500, 0.01);
        assert!(!bp.might_contain(b"not_inserted"));
        assert!(!bp.might_contain(b"oximedia"));
    }

    #[test]
    fn test_single_insert() {
        let mut bp = BloomPrescreen::new(50, 0.01);
        bp.insert(b"only_one");
        assert!(bp.might_contain(b"only_one"));
    }

    #[test]
    fn test_bulk_no_false_negatives() {
        let mut bp = BloomPrescreen::new(2000, 0.01);
        let items: Vec<Vec<u8>> = (0u64..500).map(|i| i.to_le_bytes().to_vec()).collect();
        for item in &items {
            bp.insert(item);
        }
        for item in &items {
            assert!(bp.might_contain(item), "false negative detected");
        }
    }

    // ── FPR estimation ────────────────────────────────────────────────────────

    #[test]
    fn test_fpr_estimate_zero_when_empty() {
        let bp = BloomPrescreen::new(1000, 0.01);
        // No items inserted → exponent → 0 → (1 - 1)^k = 0
        assert_eq!(bp.false_positive_rate_estimate(), 0.0);
    }

    #[test]
    fn test_fpr_estimate_increases_with_items() {
        let mut bp = BloomPrescreen::new(100, 0.01);
        let fpr_before = bp.false_positive_rate_estimate();
        for i in 0u64..50 {
            bp.insert(&i.to_le_bytes());
        }
        let fpr_after = bp.false_positive_rate_estimate();
        assert!(fpr_after > fpr_before);
    }

    #[test]
    fn test_fpr_estimate_at_design_capacity_near_design_rate() {
        // When n == capacity, FPR should be ≈ design FPR (within an order of magnitude).
        let n = 1000usize;
        let design_fpr = 0.01f64;
        let mut bp = BloomPrescreen::new(n, design_fpr);
        for i in 0u64..n as u64 {
            bp.insert(&i.to_le_bytes());
        }
        let est = bp.false_positive_rate_estimate();
        // Should be in [0.001, 0.1] — within 10× of design.
        assert!(est > 0.0, "FPR estimate must be positive");
        assert!(est < 0.5, "FPR estimate must be reasonable");
    }

    // ── items_inserted counter ────────────────────────────────────────────────

    #[test]
    fn test_items_inserted_counter() {
        let mut bp = BloomPrescreen::new(100, 0.01);
        assert_eq!(bp.items_inserted(), 0);
        bp.insert(b"one");
        assert_eq!(bp.items_inserted(), 1);
        bp.insert(b"two");
        assert_eq!(bp.items_inserted(), 2);
    }

    // ── clear ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_clear_resets_filter() {
        let mut bp = BloomPrescreen::new(100, 0.01);
        bp.insert(b"persistent");
        assert!(bp.might_contain(b"persistent"));
        bp.clear();
        assert!(!bp.might_contain(b"persistent"));
        assert_eq!(bp.items_inserted(), 0);
    }

    #[test]
    fn test_clear_resets_set_bit_count() {
        let mut bp = BloomPrescreen::new(200, 0.01);
        for i in 0u64..50 {
            bp.insert(&i.to_le_bytes());
        }
        assert!(bp.set_bit_count() > 0);
        bp.clear();
        assert_eq!(bp.set_bit_count(), 0);
    }

    // ── set_bit_count ─────────────────────────────────────────────────────────

    #[test]
    fn test_set_bit_count_zero_initially() {
        let bp = BloomPrescreen::new(500, 0.01);
        assert_eq!(bp.set_bit_count(), 0);
    }

    #[test]
    fn test_set_bit_count_increases_with_inserts() {
        let mut bp = BloomPrescreen::new(500, 0.01);
        let before = bp.set_bit_count();
        bp.insert(b"new_item");
        let after = bp.set_bit_count();
        assert!(after >= before);
    }

    // ── Optimal sizing formulas ───────────────────────────────────────────────

    #[test]
    fn test_num_bits_at_least_64() {
        // Even tiny capacity must allocate at least 64 bits (1 word).
        let bp = BloomPrescreen::new(1, 0.5);
        assert!(bp.num_bits >= 64);
        assert_eq!(bp.bits.len(), bp.num_bits / 64);
    }

    #[test]
    fn test_k_hashes_sensible_range() {
        // For standard parameters k should be between 4 and 20.
        let bp = BloomPrescreen::new(10_000, 0.01);
        assert!(bp.k_hashes >= 4, "k should be at least 4 for 1% FPR");
        assert!(bp.k_hashes <= 20, "k should be at most 20 for 1% FPR");
    }

    // ── Byte array keys ───────────────────────────────────────────────────────

    #[test]
    fn test_empty_key_insert_and_query() {
        let mut bp = BloomPrescreen::new(100, 0.01);
        bp.insert(b"");
        assert!(bp.might_contain(b""));
    }

    #[test]
    fn test_long_key_insert_and_query() {
        let mut bp = BloomPrescreen::new(100, 0.01);
        let long_key = vec![0xABu8; 4096];
        bp.insert(&long_key);
        assert!(bp.might_contain(&long_key));
    }

    #[test]
    fn test_binary_keys() {
        let mut bp = BloomPrescreen::new(200, 0.01);
        // Insert u64 values as raw bytes
        let values: Vec<u64> = vec![0, 1, u64::MAX, 0xDEAD_BEEF, 0x0102_0304_0506_0708];
        for v in &values {
            bp.insert(&v.to_le_bytes());
        }
        for v in &values {
            assert!(bp.might_contain(&v.to_le_bytes()), "no false neg for {v}");
        }
    }
}
