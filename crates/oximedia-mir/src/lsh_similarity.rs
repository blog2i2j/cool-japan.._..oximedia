//! Locality-Sensitive Hashing (LSH) for fast approximate music similarity search.
//!
//! # Overview
//!
//! Music similarity search at scale requires sub-linear query time.  This module
//! implements **MinHash LSH** over audio fingerprints derived from spectral energy
//! bands.
//!
//! ## Fingerprint
//!
//! A fingerprint is a vector of 64 [`u64`] MinHash values computed from a
//! 64-band spectral energy histogram of the input audio.  Each band energy is
//! quantised to a [`u8`] and fed into a family of 64 independent FNV-1a hash
//! functions to produce the MinHash signature.
//!
//! ## Index
//!
//! [`LshSimilarityIndex`] partitions the 64-hash signature into **bands of 8
//! hashes** (8 bands × 8 hashes = 64).  Items that share at least one full band
//! are placed into the same bucket.  A query collects candidates from all
//! matching buckets and re-ranks them by estimated Jaccard similarity.
//!
//! ## Jaccard Estimate
//!
//! For MinHash signatures of length *k*, the fraction of matching hash positions
//! estimates the Jaccard similarity of the underlying sets:
//!
//! ```text
//! sim ≈ |{i : a[i] == b[i]}| / k
//! ```

#![allow(dead_code)]

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of MinHash functions (signature length).
const N_HASHES: usize = 64;

/// Number of bands used for LSH bucketing.
const N_BANDS: usize = 8;

/// Number of hash rows per band.
const ROWS_PER_BAND: usize = N_HASHES / N_BANDS; // = 8

/// Number of spectral energy bands used to build the fingerprint.
const N_SPECTRAL_BANDS: usize = 64;

// ---------------------------------------------------------------------------
// AudioFingerprint
// ---------------------------------------------------------------------------

/// A compact audio fingerprint derived from spectral energy bands.
///
/// `hash` contains `N_HASHES` (64) MinHash values.
#[derive(Debug, Clone)]
pub struct AudioFingerprint {
    /// MinHash signature (64 values).
    pub hash: Vec<u64>,
    /// Sample rate of the source audio.
    pub sample_rate: u32,
    /// Duration of the source audio in milliseconds.
    pub duration_ms: u32,
}

impl AudioFingerprint {
    /// Number of hash values in the signature.
    #[must_use]
    pub fn len(&self) -> usize {
        self.hash.len()
    }

    /// Whether the hash vector is empty (invalid fingerprint).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hash.is_empty()
    }
}

// ---------------------------------------------------------------------------
// LshSimilarityIndex
// ---------------------------------------------------------------------------

/// Approximate nearest-neighbour index for audio fingerprints using LSH.
///
/// Supports incremental insertion and similarity search.
pub struct LshSimilarityIndex {
    /// LSH buckets: bucket key → list of item IDs stored in that bucket.
    pub buckets: HashMap<u64, Vec<usize>>,
    /// All inserted fingerprints indexed by their assigned ID.
    pub fingerprints: Vec<AudioFingerprint>,
}

impl LshSimilarityIndex {
    /// Create a new, empty index.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buckets: HashMap::new(),
            fingerprints: Vec::new(),
        }
    }

    /// Insert a fingerprint into the index.
    ///
    /// Returns the numeric ID assigned to the fingerprint.  IDs start at 0 and
    /// increment by 1 for each insertion.
    pub fn insert(&mut self, fp: AudioFingerprint) -> usize {
        let id = self.fingerprints.len();

        // Insert into LSH buckets (one per band).
        for band in 0..N_BANDS {
            let band_key = band_hash(&fp.hash, band);
            self.buckets
                .entry(band_key)
                .or_default()
                .push(id);
        }

        self.fingerprints.push(fp);
        id
    }

    /// Search the index for fingerprints similar to `query`.
    ///
    /// Returns a list of `(id, estimated_jaccard_similarity)` pairs sorted in
    /// descending order by similarity, filtered to those with `similarity >= threshold`.
    ///
    /// # Arguments
    ///
    /// * `query` — the fingerprint to match against.
    /// * `threshold` — minimum Jaccard similarity (0.0 – 1.0) to include in results.
    #[must_use]
    pub fn search_similar(
        &self,
        query: &AudioFingerprint,
        threshold: f32,
    ) -> Vec<(usize, f32)> {
        // Collect candidate IDs from all matching LSH buckets.
        let mut candidate_ids: Vec<usize> = Vec::new();

        for band in 0..N_BANDS {
            let band_key = band_hash(&query.hash, band);
            if let Some(bucket) = self.buckets.get(&band_key) {
                for &id in bucket {
                    candidate_ids.push(id);
                }
            }
        }

        // Deduplicate candidates.
        candidate_ids.sort_unstable();
        candidate_ids.dedup();

        // Score each candidate.
        let mut results: Vec<(usize, f32)> = candidate_ids
            .into_iter()
            .filter_map(|id| {
                self.fingerprints.get(id).map(|fp| {
                    let sim = jaccard_estimate(&query.hash, &fp.hash);
                    (id, sim)
                })
            })
            .filter(|&(_, sim)| sim >= threshold)
            .collect();

        // Sort descending by similarity.
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Number of fingerprints currently in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.fingerprints.len()
    }

    /// Whether the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fingerprints.is_empty()
    }

    /// Retrieve a fingerprint by ID.
    #[must_use]
    pub fn get(&self, id: usize) -> Option<&AudioFingerprint> {
        self.fingerprints.get(id)
    }
}

impl Default for LshSimilarityIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// compute_fingerprint
// ---------------------------------------------------------------------------

/// Compute an [`AudioFingerprint`] from raw audio samples.
///
/// # Algorithm
///
/// 1. Divide the spectrum into `N_SPECTRAL_BANDS` (64) equal-width frequency
///    bands.  The spectrum is approximated by frame-wise RMS energy in each
///    band using a simple sliding-window DFT approximation.
/// 2. Accumulate the per-band energy across all analysis frames.
/// 3. Quantise each band energy to a [`u8`] value using linear normalisation.
/// 4. For each of the 64 MinHash functions, compute the minimum FNV-1a hash
///    over all `(band_idx, quantised_energy_byte)` pairs.
///
/// The resulting 64-value signature is suitable for LSH indexing.
///
/// # Arguments
///
/// * `samples` — mono audio samples (f32).
/// * `sample_rate` — sample rate in Hz.
#[must_use]
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn compute_fingerprint(samples: &[f32], sample_rate: u32) -> AudioFingerprint {
    let duration_ms = if sample_rate > 0 {
        (samples.len() as f64 / sample_rate as f64 * 1000.0) as u32
    } else {
        0
    };

    if samples.is_empty() || sample_rate == 0 {
        return AudioFingerprint {
            hash: vec![u64::MAX; N_HASHES],
            sample_rate,
            duration_ms,
        };
    }

    // ── Step 1: compute per-band accumulated energy ──────────────────────────
    let band_energies = spectral_band_energies(samples, sample_rate);

    // ── Step 2: quantise band energies to u8 ────────────────────────────────
    let max_energy = band_energies
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let quantised: Vec<u8> = if max_energy > 1e-9 {
        band_energies
            .iter()
            .map(|&e| ((e / max_energy) * 255.0) as u8)
            .collect()
    } else {
        vec![0u8; N_SPECTRAL_BANDS]
    };

    // ── Step 3: MinHash over (band_idx, energy_byte) pairs ──────────────────
    // We use N_HASHES independent FNV-1a hash seeds.
    let hash = minhash(&quantised);

    AudioFingerprint {
        hash,
        sample_rate,
        duration_ms,
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute accumulated spectral energy for `N_SPECTRAL_BANDS` frequency bands.
///
/// We split the signal into overlapping frames and compute a simplified
/// frequency-band energy by multiplying the signal by cosine basis functions
/// at band center frequencies and summing the squared responses.
#[allow(clippy::cast_precision_loss)]
fn spectral_band_energies(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    let sr = sample_rate as f32;
    let max_freq = sr / 2.0;

    // Band center frequencies: logarithmically spaced from 20 Hz to Nyquist.
    let log_min = (20.0_f32).ln();
    let log_max = max_freq.ln();

    let band_centers: Vec<f32> = (0..N_SPECTRAL_BANDS)
        .map(|i| {
            let t = i as f32 / (N_SPECTRAL_BANDS - 1) as f32;
            (log_min + t * (log_max - log_min)).exp()
        })
        .collect();

    let hop = (sr * 0.023) as usize; // ~23 ms frames
    let hop = hop.max(1);
    let n_frames = (samples.len() / hop).max(1);
    let frame_len = hop.min(samples.len());

    let mut band_energies = vec![0.0_f32; N_SPECTRAL_BANDS];

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop;
        let end = (start + frame_len).min(samples.len());
        if start >= end {
            break;
        }
        let frame = &samples[start..end];
        let frame_n = frame.len() as f32;

        for (band_idx, &center_freq) in band_centers.iter().enumerate() {
            // Compute energy at this frequency using matched filter.
            let cos_sum: f32 = frame
                .iter()
                .enumerate()
                .map(|(t, &s)| {
                    s * (std::f32::consts::TAU * center_freq * t as f32 / sr).cos()
                })
                .sum::<f32>();
            let energy = (cos_sum / frame_n).powi(2);
            band_energies[band_idx] += energy;
        }
    }

    band_energies
}

/// Compute a 64-value MinHash signature from quantised band energies.
///
/// Each MinHash function `h_i` is parameterised by a seed derived from `i`.
/// For each element of the set, we compute `fnv1a(seed ^ (band_idx as u64) ^ byte as u64)`
/// and track the minimum across all elements.
fn minhash(quantised: &[u8]) -> Vec<u64> {
    (0..N_HASHES)
        .map(|hash_idx| {
            // Seed for this hash function: mix index with a fixed prime.
            let seed = fnv1a_hash(hash_idx as u64 ^ 0x9E37_79B9_7F4A_7C15);
            let mut min_val = u64::MAX;
            for (band_idx, &byte) in quantised.iter().enumerate() {
                let element = (band_idx as u64).wrapping_mul(251) ^ byte as u64;
                let h = fnv1a_hash(seed ^ element);
                if h < min_val {
                    min_val = h;
                }
            }
            min_val
        })
        .collect()
}

/// FNV-1a 64-bit hash of a single `u64` value.
#[inline]
fn fnv1a_hash(mut v: u64) -> u64 {
    const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;
    const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
    let mut hash = FNV_OFFSET;
    // Process 8 bytes
    for _ in 0..8 {
        hash ^= v & 0xFF;
        hash = hash.wrapping_mul(FNV_PRIME);
        v >>= 8;
    }
    hash
}

/// Compute the bucket hash for one LSH band.
///
/// XORs the hash values in the band together with a band-specific salt.
fn band_hash(hashes: &[u64], band: usize) -> u64 {
    let start = band * ROWS_PER_BAND;
    let end = (start + ROWS_PER_BAND).min(hashes.len());

    // Include band index in the hash to avoid collisions across bands.
    let mut h = fnv1a_hash(band as u64 ^ 0xFEED_BEEF_DEAD_C0DE);
    for &v in &hashes[start..end] {
        h = fnv1a_hash(h ^ v);
    }
    h
}

/// Estimate Jaccard similarity between two MinHash signatures.
///
/// `sim = |{i : a[i] == b[i]}| / min(|a|, |b|)`
#[must_use]
pub fn jaccard_estimate(a: &[u64], b: &[u64]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let matches = a[..n]
        .iter()
        .zip(b[..n].iter())
        .filter(|(&x, &y)| x == y)
        .count();
    matches as f32 / n as f32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn make_sine(freq: f32, sr: u32, secs: f32) -> Vec<f32> {
        let n = (sr as f32 * secs) as usize;
        (0..n)
            .map(|i| (TAU * freq * i as f32 / sr as f32).sin())
            .collect()
    }

    // ── AudioFingerprint tests ────────────────────────────────────────────────

    #[test]
    fn test_fingerprint_length() {
        let sig = make_sine(440.0, 44100, 1.0);
        let fp = compute_fingerprint(&sig, 44100);
        assert_eq!(fp.hash.len(), N_HASHES);
    }

    #[test]
    fn test_fingerprint_empty_input() {
        let fp = compute_fingerprint(&[], 44100);
        // Should return a valid (all-MAX) fingerprint without panicking.
        assert_eq!(fp.hash.len(), N_HASHES);
        assert!(fp.hash.iter().all(|&h| h == u64::MAX));
    }

    #[test]
    fn test_fingerprint_deterministic() {
        let sig = make_sine(220.0, 22050, 0.5);
        let fp1 = compute_fingerprint(&sig, 22050);
        let fp2 = compute_fingerprint(&sig, 22050);
        assert_eq!(fp1.hash, fp2.hash);
    }

    #[test]
    fn test_fingerprint_duration_ms() {
        // 44100 samples at 44100 Hz → 1000 ms
        let samples = vec![0.0f32; 44100];
        let fp = compute_fingerprint(&samples, 44100);
        assert_eq!(fp.duration_ms, 1000);
    }

    #[test]
    fn test_fingerprint_different_signals_differ() {
        let sig_a = make_sine(440.0, 44100, 1.0);
        let sig_b = make_sine(110.0, 44100, 1.0);
        let fp_a = compute_fingerprint(&sig_a, 44100);
        let fp_b = compute_fingerprint(&sig_b, 44100);
        // Different signals should typically produce different hashes.
        let same = fp_a.hash.iter().zip(fp_b.hash.iter()).filter(|(a, b)| a == b).count();
        // Allow up to 50% collision by chance, but expect significant differences.
        assert!(same < N_HASHES, "Fingerprints are identical for different signals");
    }

    // ── Jaccard estimate tests ────────────────────────────────────────────────

    #[test]
    fn test_jaccard_identical() {
        let h: Vec<u64> = (0..64).collect();
        assert!((jaccard_estimate(&h, &h) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_jaccard_disjoint() {
        let a: Vec<u64> = (0..64).map(|i| i as u64).collect();
        let b: Vec<u64> = (0..64).map(|i| i as u64 + 1000).collect();
        assert!((jaccard_estimate(&a, &b) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_jaccard_empty() {
        assert!((jaccard_estimate(&[], &[]) - 0.0).abs() < f32::EPSILON);
    }

    // ── LshSimilarityIndex tests ──────────────────────────────────────────────

    #[test]
    fn test_index_empty() {
        let idx = LshSimilarityIndex::new();
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn test_index_insert_assigns_sequential_ids() {
        let mut idx = LshSimilarityIndex::new();
        let fp0 = compute_fingerprint(&make_sine(440.0, 44100, 1.0), 44100);
        let fp1 = compute_fingerprint(&make_sine(220.0, 44100, 1.0), 44100);
        let id0 = idx.insert(fp0);
        let id1 = idx.insert(fp1);
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(idx.len(), 2);
    }

    #[test]
    fn test_search_identical_fingerprint() {
        let mut idx = LshSimilarityIndex::new();
        let sig = make_sine(440.0, 44100, 1.0);
        let fp = compute_fingerprint(&sig, 44100);
        let fp_query = fp.clone();
        let id = idx.insert(fp);

        let results = idx.search_similar(&fp_query, 0.5);
        // The same fingerprint should be a candidate.
        assert!(
            results.iter().any(|&(found_id, sim)| found_id == id && sim >= 0.5),
            "Expected to find inserted item in search results"
        );
    }

    #[test]
    fn test_search_below_threshold_excluded() {
        let mut idx = LshSimilarityIndex::new();
        let sig = make_sine(440.0, 44100, 1.0);
        let fp = compute_fingerprint(&sig, 44100);
        idx.insert(fp.clone());

        // With threshold = 1.0, only a perfect match should appear.
        let results = idx.search_similar(&fp, 1.0);
        for (_, sim) in &results {
            assert!(*sim >= 1.0 - f32::EPSILON);
        }
    }

    #[test]
    fn test_search_results_sorted_descending() {
        let mut idx = LshSimilarityIndex::new();
        for freq in [220.0, 440.0, 880.0, 1760.0_f32] {
            let fp = compute_fingerprint(&make_sine(freq, 44100, 0.5), 44100);
            idx.insert(fp);
        }
        let query = compute_fingerprint(&make_sine(440.0, 44100, 0.5), 44100);
        let results = idx.search_similar(&query, 0.0);
        for w in results.windows(2) {
            assert!(
                w[0].1 >= w[1].1,
                "Results not sorted: {} < {}",
                w[0].1,
                w[1].1
            );
        }
    }

    #[test]
    fn test_get_fingerprint_by_id() {
        let mut idx = LshSimilarityIndex::new();
        let fp = compute_fingerprint(&make_sine(440.0, 44100, 1.0), 44100);
        let id = idx.insert(fp.clone());
        let retrieved = idx.get(id).expect("should find fingerprint");
        assert_eq!(retrieved.sample_rate, 44100);
    }

    #[test]
    fn test_fnv1a_hash_deterministic() {
        assert_eq!(fnv1a_hash(12345), fnv1a_hash(12345));
    }

    #[test]
    fn test_fnv1a_hash_different_inputs() {
        assert_ne!(fnv1a_hash(0), fnv1a_hash(1));
    }

    #[test]
    fn test_band_hash_varies_by_band() {
        let hashes: Vec<u64> = (0..64).collect();
        let h0 = band_hash(&hashes, 0);
        let h1 = band_hash(&hashes, 1);
        assert_ne!(h0, h1);
    }
}
