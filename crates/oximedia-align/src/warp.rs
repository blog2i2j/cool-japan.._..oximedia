//! Time warp / time-stretching alignment using Dynamic Time Warping (DTW).
//!
//! Provides:
//! - [`DtwAligner`] – standard DTW with full cost matrix and Euclidean distance.
//! - [`WarpPath`] – the DTW alignment path and timestamp remapping utilities.
//! - [`WarpCurve`] – a continuous (`time_ms`, `offset_ms`) curve derived from a path.
//! - [`WarpSmoothing`] – moving-average smoother for warp curves.

#![allow(dead_code)]

/// A DTW alignment path represented as matched index pairs `(i_a, i_b)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WarpPath {
    /// Ordered list of aligned index pairs from sequence A and sequence B.
    pub pairs: Vec<(usize, usize)>,
}

impl WarpPath {
    /// Create a warp path from a list of index pairs.
    #[must_use]
    pub fn new(pairs: Vec<(usize, usize)>) -> Self {
        Self { pairs }
    }

    /// Remap a set of timestamps (in milliseconds) from sequence A's time axis
    /// to sequence B's time axis.
    ///
    /// For each timestamp in `original_ms`, the method finds the closest index
    /// in the path and returns the corresponding B-index scaled by the per-frame
    /// duration.
    ///
    /// # Arguments
    /// * `original_ms` – timestamps in A's coordinate system (monotonically
    ///   increasing, same unit as frame indices × frame duration).
    ///
    /// The caller is responsible for choosing a consistent unit (e.g. 1 index =
    /// 1 ms, or use [`WarpCurve`] for fractional frame rates).
    #[must_use]
    pub fn apply_to_timestamps(&self, original_ms: &[u64]) -> Vec<u64> {
        if self.pairs.is_empty() || original_ms.is_empty() {
            return original_ms.to_vec();
        }

        original_ms
            .iter()
            .map(|&t| {
                // Find the pair whose A-index is closest to t.
                let closest = self
                    .pairs
                    .iter()
                    .min_by_key(|(ia, _)| (*ia as i64 - t as i64).unsigned_abs())
                    .copied()
                    .unwrap_or((0, 0));
                closest.1 as u64
            })
            .collect()
    }

    /// Return the length of the path.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Return `true` if the path is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }
}

/// Dynamic Time Warping aligner.
///
/// Uses the standard full-matrix DTW algorithm with Euclidean (absolute value)
/// distance between scalar samples.
pub struct DtwAligner;

impl DtwAligner {
    /// Create a new DTW aligner.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Compute the DTW distance and alignment path between two sequences.
    ///
    /// Returns `(distance, path)` where `distance` is the normalised DTW cost
    /// (divided by path length) and `path` contains the matched index pairs.
    ///
    /// # Panics
    /// Does not panic; returns empty path and `0.0` distance for empty inputs.
    #[must_use]
    pub fn compute(seq_a: &[f32], seq_b: &[f32]) -> (f32, WarpPath) {
        let na = seq_a.len();
        let nb = seq_b.len();

        if na == 0 || nb == 0 {
            return (0.0, WarpPath::new(vec![]));
        }

        // Build the DTW cost matrix (na × nb).
        let inf = f32::INFINITY;
        let mut dtw = vec![vec![inf; nb]; na];

        dtw[0][0] = (seq_a[0] - seq_b[0]).abs();

        for j in 1..nb {
            dtw[0][j] = dtw[0][j - 1] + (seq_a[0] - seq_b[j]).abs();
        }
        for i in 1..na {
            dtw[i][0] = dtw[i - 1][0] + (seq_a[i] - seq_b[0]).abs();
        }
        for i in 1..na {
            for j in 1..nb {
                let cost = (seq_a[i] - seq_b[j]).abs();
                let min_prev = dtw[i - 1][j].min(dtw[i][j - 1]).min(dtw[i - 1][j - 1]);
                dtw[i][j] = cost + min_prev;
            }
        }

        // Back-track to recover the path.
        let mut path = Vec::new();
        let mut i = na - 1;
        let mut j = nb - 1;
        path.push((i, j));

        while i > 0 || j > 0 {
            if i == 0 {
                j -= 1;
            } else if j == 0 {
                i -= 1;
            } else {
                let diag = dtw[i - 1][j - 1];
                let left = dtw[i][j - 1];
                let up = dtw[i - 1][j];
                if diag <= left && diag <= up {
                    i -= 1;
                    j -= 1;
                } else if left < up {
                    j -= 1;
                } else {
                    i -= 1;
                }
            }
            path.push((i, j));
        }

        path.reverse();

        let total_cost = dtw[na - 1][nb - 1];
        let norm_cost = if path.is_empty() {
            0.0
        } else {
            total_cost / path.len() as f32
        };

        (norm_cost, WarpPath::new(path))
    }
}

impl Default for DtwAligner {
    fn default() -> Self {
        Self::new()
    }
}

/// A continuous warp curve mapping original timestamps (ms) to signed offsets
/// (ms).  Each point is `(original_ms, offset_ms)`.
#[derive(Debug, Clone)]
pub struct WarpCurve {
    /// Ordered control points: `(original_time_ms, offset_ms)`.
    pub points: Vec<(u64, i64)>,
}

impl WarpCurve {
    /// Create a warp curve from a [`WarpPath`] and a frames-per-second value.
    ///
    /// Each path pair `(ia, ib)` is converted: the A-time is `ia * frame_ms`
    /// and the offset is `(ib as i64 - ia as i64) * frame_ms`.
    #[must_use]
    pub fn from_path(path: &WarpPath, fps: f32) -> Self {
        if fps <= 0.0 || path.is_empty() {
            return Self { points: vec![] };
        }

        let frame_ms = (1000.0 / fps) as i64;
        let mut points: Vec<(u64, i64)> = path
            .pairs
            .iter()
            .map(|&(ia, ib)| {
                let t = ia as u64 * frame_ms as u64;
                let offset = (ib as i64 - ia as i64) * frame_ms;
                (t, offset)
            })
            .collect();

        // Deduplicate by time, keeping the last (should already be monotone).
        points.dedup_by_key(|p| p.0);
        Self { points }
    }

    /// Linearly interpolate the offset at `time_ms`.
    ///
    /// Clamps to the first/last point outside the curve's range.
    #[must_use]
    pub fn interpolate(&self, time_ms: u64) -> i64 {
        if self.points.is_empty() {
            return 0;
        }
        if time_ms <= self.points[0].0 {
            return self.points[0].1;
        }
        let last = self.points[self.points.len() - 1];
        if time_ms >= last.0 {
            return last.1;
        }

        // Binary-search for the surrounding segment.
        let idx = self.points.partition_point(|&(t, _)| t <= time_ms);
        let (t0, o0) = self.points[idx - 1];
        let (t1, o1) = self.points[idx];

        let alpha = (time_ms - t0) as f64 / (t1 - t0) as f64;
        let interpolated = o0 as f64 + alpha * (o1 as f64 - o0 as f64);
        interpolated.round() as i64
    }
}

/// Moving-average smoother for [`WarpCurve`]s.
pub struct WarpSmoothing;

impl WarpSmoothing {
    /// Smooth a warp curve using a symmetric moving average of `window` samples.
    ///
    /// Points at the boundaries use a reduced window (causal/anticausal
    /// clamping).
    #[must_use]
    pub fn smooth(curve: &WarpCurve, window: usize) -> WarpCurve {
        let n = curve.points.len();
        if n == 0 || window <= 1 {
            return curve.clone();
        }

        let half = window / 2;
        let smoothed_points: Vec<(u64, i64)> = (0..n)
            .map(|i| {
                let start = i.saturating_sub(half);
                let end = (i + half + 1).min(n);
                let count = end - start;
                let sum: i64 = curve.points[start..end].iter().map(|p| p.1).sum();
                let avg = (sum as f64 / count as f64).round() as i64;
                (curve.points[i].0, avg)
            })
            .collect();

        WarpCurve {
            points: smoothed_points,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── DtwAligner ────────────────────────────────────────────────────────────

    #[test]
    fn test_dtw_empty_inputs() {
        let (dist, path) = DtwAligner::compute(&[], &[1.0]);
        assert_eq!(dist, 0.0);
        assert!(path.is_empty());

        let (dist2, path2) = DtwAligner::compute(&[1.0], &[]);
        assert_eq!(dist2, 0.0);
        assert!(path2.is_empty());
    }

    #[test]
    fn test_dtw_identical_sequences() {
        let seq = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let (dist, path) = DtwAligner::compute(&seq, &seq);
        assert_eq!(
            dist, 0.0,
            "identical sequences should have zero DTW distance"
        );
        // Path should be diagonal.
        for (i, &(ia, ib)) in path.pairs.iter().enumerate() {
            let _ = i;
            assert_eq!(ia, ib, "diagonal path expected for identical sequences");
        }
    }

    #[test]
    fn test_dtw_shifted_sequence() {
        // seq_b is seq_a shifted by one; DTW should find a near-zero alignment.
        let seq_a = vec![0.0f32, 1.0, 2.0, 3.0, 4.0];
        let seq_b = vec![0.0f32, 0.0, 1.0, 2.0, 3.0, 4.0];
        let (dist, path) = DtwAligner::compute(&seq_a, &seq_b);
        assert!(
            dist < 1.0,
            "shifted sequence should have low DTW distance: {dist}"
        );
        assert!(!path.is_empty());
    }

    #[test]
    fn test_dtw_path_starts_at_origin_ends_at_corner() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![1.0f32, 2.5, 3.0, 3.5];
        let (_, path) = DtwAligner::compute(&a, &b);
        assert_eq!(path.pairs[0], (0, 0), "path must start at (0,0)");
        let last = *path.pairs.last().expect("last should be valid");
        assert_eq!(
            last,
            (a.len() - 1, b.len() - 1),
            "path must end at (na-1, nb-1)"
        );
    }

    #[test]
    fn test_dtw_single_elements() {
        let (dist, path) = DtwAligner::compute(&[3.0], &[5.0]);
        assert!((dist - 2.0).abs() < 1e-6);
        assert_eq!(path.pairs, vec![(0, 0)]);
    }

    // ── WarpPath ─────────────────────────────────────────────────────────────

    #[test]
    fn test_warp_path_apply_timestamps_empty_path() {
        let path = WarpPath::new(vec![]);
        let ts = vec![100u64, 200, 300];
        let result = path.apply_to_timestamps(&ts);
        assert_eq!(result, ts);
    }

    #[test]
    fn test_warp_path_apply_timestamps() {
        // Path: A[0]→B[0], A[1]→B[2], A[2]→B[3]
        let path = WarpPath::new(vec![(0, 0), (1, 2), (2, 3)]);
        // timestamp 1 → closest A-index 1 → B-index 2
        let result = path.apply_to_timestamps(&[1]);
        assert_eq!(result, vec![2]);
    }

    #[test]
    fn test_warp_path_len() {
        let path = WarpPath::new(vec![(0, 0), (1, 1), (2, 2)]);
        assert_eq!(path.len(), 3);
        assert!(!path.is_empty());
    }

    // ── WarpCurve ─────────────────────────────────────────────────────────────

    #[test]
    fn test_warp_curve_from_path_empty() {
        let path = WarpPath::new(vec![]);
        let curve = WarpCurve::from_path(&path, 25.0);
        assert!(curve.points.is_empty());
    }

    #[test]
    fn test_warp_curve_from_path_diagonal() {
        // Diagonal path → all offsets zero.
        let pairs: Vec<(usize, usize)> = (0..5).map(|i| (i, i)).collect();
        let path = WarpPath::new(pairs);
        let curve = WarpCurve::from_path(&path, 25.0);
        for &(_, offset) in &curve.points {
            assert_eq!(offset, 0, "diagonal path should produce zero offsets");
        }
    }

    #[test]
    fn test_warp_curve_interpolate_clamp() {
        let curve = WarpCurve {
            points: vec![(0, 10), (1000, 20)],
        };
        assert_eq!(curve.interpolate(0), 10);
        assert_eq!(curve.interpolate(2000), 20); // clamp to last
    }

    #[test]
    fn test_warp_curve_interpolate_midpoint() {
        let curve = WarpCurve {
            points: vec![(0, 0), (1000, 100)],
        };
        let mid = curve.interpolate(500);
        assert!(
            (mid - 50).abs() <= 1,
            "midpoint offset should be ~50, got {mid}"
        );
    }

    // ── WarpSmoothing ─────────────────────────────────────────────────────────

    #[test]
    fn test_warp_smoothing_constant_curve() {
        let curve = WarpCurve {
            points: vec![(0, 5), (100, 5), (200, 5), (300, 5)],
        };
        let smoothed = WarpSmoothing::smooth(&curve, 3);
        for &(_, v) in &smoothed.points {
            assert_eq!(
                v, 5,
                "constant curve should remain unchanged after smoothing"
            );
        }
    }

    #[test]
    fn test_warp_smoothing_reduces_spike() {
        let curve = WarpCurve {
            points: vec![(0, 0), (100, 0), (200, 100), (300, 0), (400, 0)],
        };
        let smoothed = WarpSmoothing::smooth(&curve, 3);
        // The spike at index 2 should be reduced.
        let spike_val = smoothed.points[2].1;
        assert!(spike_val < 100, "spike should be attenuated: {spike_val}");
    }

    #[test]
    fn test_warp_smoothing_empty() {
        let curve = WarpCurve { points: vec![] };
        let smoothed = WarpSmoothing::smooth(&curve, 5);
        assert!(smoothed.points.is_empty());
    }
}
