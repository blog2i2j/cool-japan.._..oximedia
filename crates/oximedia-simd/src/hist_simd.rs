//! SIMD-friendly histogram computation for luma (grayscale) buffers.
//!
//! This module provides [`compute_histogram_fast`], which accumulates a
//! 256-bucket histogram over a byte slice.  The inner loop is manually unrolled
//! 4× so that the compiler can schedule independent load/increment chains,
//! keeping the hardware busy even on in-order cores.
//!
//! For larger buffers a multi-accumulator strategy is used: four independent
//! 256-element `u32` arrays are updated in lock-step and summed at the end.
//! This avoids the read-after-write hazard on the histogram bucket that would
//! otherwise serialize the loop.

// ─── compute_histogram_fast ───────────────────────────────────────────────────

/// Compute the 256-bucket histogram of a luma (single-channel u8) buffer.
///
/// The algorithm uses a 4× unrolled loop with four independent accumulator
/// arrays to mitigate read-after-write latency on histogram buckets.  The
/// final histogram is the element-wise sum of the four accumulators.
///
/// # Parameters
///
/// * `luma` — input byte buffer (any length; may be empty)
///
/// # Returns
///
/// A `[u32; 256]` histogram where `hist[v]` counts the number of bytes with
/// value `v`.
#[must_use]
pub fn compute_histogram_fast(luma: &[u8]) -> [u32; 256] {
    // Four independent accumulators to break data-dependency chains
    let mut h0 = [0u32; 256];
    let mut h1 = [0u32; 256];
    let mut h2 = [0u32; 256];
    let mut h3 = [0u32; 256];

    let chunks = luma.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        // Safety: `chunks_exact(4)` guarantees exactly 4 bytes
        let b0 = chunk[0] as usize;
        let b1 = chunk[1] as usize;
        let b2 = chunk[2] as usize;
        let b3 = chunk[3] as usize;
        h0[b0] += 1;
        h1[b1] += 1;
        h2[b2] += 1;
        h3[b3] += 1;
    }

    // Handle trailing bytes (0–3)
    for &b in remainder {
        h0[b as usize] += 1;
    }

    // Merge the four accumulators
    let mut result = [0u32; 256];
    for i in 0..256 {
        result[i] = h0[i] + h1[i] + h2[i] + h3[i];
    }
    result
}

// ─── Weighted histogram ───────────────────────────────────────────────────────

/// Compute a weighted 256-bucket histogram.
///
/// Each element in `luma` contributes `weights[i]` counts to its bucket
/// instead of 1.  `luma` and `weights` must have the same length; extra
/// weight elements are ignored and missing ones are treated as weight 1.
#[must_use]
pub fn compute_histogram_weighted(luma: &[u8], weights: &[u32]) -> [u32; 256] {
    let mut hist = [0u32; 256];
    for (i, &b) in luma.iter().enumerate() {
        let w = weights.get(i).copied().unwrap_or(1);
        hist[b as usize] = hist[b as usize].saturating_add(w);
    }
    hist
}

// ─── Cumulative distribution ──────────────────────────────────────────────────

/// Compute the cumulative distribution function (CDF) from a histogram.
///
/// `cdf[k]` = sum of `hist[0..=k]`.  The final element `cdf[255]` equals
/// the total number of samples.
#[must_use]
pub fn histogram_to_cdf(hist: &[u32; 256]) -> [u32; 256] {
    let mut cdf = [0u32; 256];
    let mut acc = 0u32;
    for (i, &h) in hist.iter().enumerate() {
        acc = acc.saturating_add(h);
        cdf[i] = acc;
    }
    cdf
}

/// Compute the mean intensity from a histogram.
///
/// Returns `None` if the histogram is empty (all-zero).
#[must_use]
pub fn histogram_mean(hist: &[u32; 256]) -> Option<f64> {
    let total: u64 = hist.iter().map(|&h| h as u64).sum();
    if total == 0 {
        return None;
    }
    let weighted_sum: u64 = hist
        .iter()
        .enumerate()
        .map(|(v, &h)| v as u64 * h as u64)
        .sum();
    Some(weighted_sum as f64 / total as f64)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── compute_histogram_fast ────────────────────────────────────────────────

    #[test]
    fn histogram_empty_input() {
        let h = compute_histogram_fast(&[]);
        for v in h {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn histogram_all_zeros() {
        let data = vec![0u8; 256];
        let h = compute_histogram_fast(&data);
        assert_eq!(h[0], 256);
        for v in &h[1..] {
            assert_eq!(*v, 0);
        }
    }

    #[test]
    fn histogram_all_255() {
        let data = vec![255u8; 100];
        let h = compute_histogram_fast(&data);
        assert_eq!(h[255], 100);
        for v in &h[..255] {
            assert_eq!(*v, 0);
        }
    }

    #[test]
    fn histogram_counts_sum_to_length() {
        let data: Vec<u8> = (0..1000u16).map(|i| (i % 256) as u8).collect();
        let h = compute_histogram_fast(&data);
        let total: u32 = h.iter().sum();
        assert_eq!(total, data.len() as u32);
    }

    #[test]
    fn histogram_single_element() {
        let data = vec![42u8];
        let h = compute_histogram_fast(&data);
        assert_eq!(h[42], 1);
        let total: u32 = h.iter().sum();
        assert_eq!(total, 1);
    }

    #[test]
    fn histogram_non_aligned_length() {
        // 101 bytes: 25 chunks of 4 + 1 remainder
        let data: Vec<u8> = (0..101u8).collect();
        let h = compute_histogram_fast(&data);
        let total: u32 = h.iter().sum();
        assert_eq!(total, 101);
    }

    #[test]
    fn histogram_uniform_distribution() {
        let data: Vec<u8> = (0..=255u8).collect();
        let h = compute_histogram_fast(&data);
        for v in h {
            assert_eq!(v, 1);
        }
    }

    // ── histogram_to_cdf ──────────────────────────────────────────────────────

    #[test]
    fn cdf_last_element_equals_total() {
        let data: Vec<u8> = vec![0, 128, 255];
        let h = compute_histogram_fast(&data);
        let cdf = histogram_to_cdf(&h);
        assert_eq!(cdf[255], 3);
    }

    #[test]
    fn cdf_monotone_non_decreasing() {
        let data: Vec<u8> = (0..=255u8).collect();
        let h = compute_histogram_fast(&data);
        let cdf = histogram_to_cdf(&h);
        for i in 1..256 {
            assert!(cdf[i] >= cdf[i - 1]);
        }
    }

    // ── histogram_mean ────────────────────────────────────────────────────────

    #[test]
    fn mean_of_uniform_is_127_5() {
        let data: Vec<u8> = (0..=255u8).collect();
        let h = compute_histogram_fast(&data);
        let mean = histogram_mean(&h).expect("non-empty");
        assert!((mean - 127.5).abs() < 0.01, "mean={mean}");
    }

    #[test]
    fn mean_empty_histogram_is_none() {
        let h = compute_histogram_fast(&[]);
        assert!(histogram_mean(&h).is_none());
    }
}
