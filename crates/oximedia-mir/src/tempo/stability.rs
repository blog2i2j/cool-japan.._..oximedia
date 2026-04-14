//! Tempo stability analysis — measures how metronomic a track is.
//!
//! [`TempoStability`] provides rich metrics derived from a sequence of
//! inter-beat intervals (IBIs) expressed in milliseconds.  The top-level
//! free function [`compute_stability`] is a convenience wrapper.

/// Detailed metrics about how steady/consistent the BPM is across a track.
#[derive(Debug, Clone)]
pub struct TempoStability {
    /// Standard deviation of detected beat intervals (0.0 = perfectly steady).
    pub interval_std_ms: f32,
    /// Coefficient of variation = std / mean (lower = more stable).
    pub cv: f32,
    /// `true` if the track uses rubato / tempo changes (`cv > 0.05`).
    pub has_tempo_variation: bool,
    /// `true` if a systematic acceleration or deceleration is detected.
    ///
    /// Detection criterion: the absolute normalised linear-regression slope
    /// of the interval sequence exceeds 0.01 (`|slope| / mean > 0.01`).
    pub has_acceleration: bool,
    /// Stability score 0.0–1.0 (1.0 = perfectly metronomic).
    ///
    /// Computed as `(1.0 - cv.min(1.0)).max(0.0)`.
    pub stability_score: f32,
}

impl TempoStability {
    /// Compute tempo stability from a sequence of beat intervals in milliseconds.
    ///
    /// Returns `None` if fewer than two intervals are supplied (not enough
    /// data to compute variance or regression).
    #[must_use]
    pub fn from_intervals(intervals_ms: &[f32]) -> Option<Self> {
        if intervals_ms.len() < 2 {
            return None;
        }

        let n = intervals_ms.len() as f32;

        // Mean interval
        let mean: f32 = intervals_ms.iter().sum::<f32>() / n;

        // Standard deviation (population std)
        let variance: f32 = intervals_ms
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / n;
        let std_dev = variance.sqrt();

        // Coefficient of variation
        let cv = if mean.abs() > f32::EPSILON {
            std_dev / mean
        } else {
            0.0
        };

        let has_tempo_variation = cv > 0.05;
        let stability_score = (1.0 - cv.min(1.0)).max(0.0);

        // Acceleration detection via linear regression on intervals.
        // slope = (sum(x*y) - n*mean_x*mean_y) / (sum(x^2) - n*mean_x^2)
        // where x is the index (0..n) and y is the interval value.
        let has_acceleration = detect_acceleration(intervals_ms, mean);

        Some(Self {
            interval_std_ms: std_dev,
            cv,
            has_tempo_variation,
            has_acceleration,
            stability_score,
        })
    }
}

/// Compute tempo stability from a sequence of beat intervals in milliseconds.
///
/// Convenience free function that delegates to [`TempoStability::from_intervals`].
///
/// Returns `None` if `intervals_ms` has fewer than 2 elements.
#[must_use]
pub fn compute_stability(intervals_ms: &[f32]) -> Option<TempoStability> {
    TempoStability::from_intervals(intervals_ms)
}

/// Detect whether there is a significant linear trend in the interval sequence.
///
/// Uses ordinary least-squares regression of interval value on index.
/// Returns `true` when the normalised absolute slope `|slope| / mean` > 0.01.
fn detect_acceleration(intervals_ms: &[f32], mean: f32) -> bool {
    let n = intervals_ms.len() as f32;
    if n < 2.0 {
        return false;
    }

    // Mean of indices
    let mean_x = (n - 1.0) / 2.0;

    // sum((x - mean_x) * (y - mean_y))  /  sum((x - mean_x)^2)
    let mut sum_xy = 0.0_f32;
    let mut sum_xx = 0.0_f32;

    for (i, &y) in intervals_ms.iter().enumerate() {
        let x = i as f32 - mean_x;
        sum_xy += x * (y - mean);
        sum_xx += x * x;
    }

    if sum_xx < f32::EPSILON {
        return false;
    }

    let slope = sum_xy / sum_xx;

    // Normalise: |slope| relative to mean interval
    if mean.abs() < f32::EPSILON {
        return false;
    }

    (slope / mean).abs() > 0.01
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build perfectly even intervals of `interval_ms` repeated `n` times.
    fn even(interval_ms: f32, n: usize) -> Vec<f32> {
        vec![interval_ms; n]
    }

    #[test]
    fn test_perfect_metronome_stability_one() {
        // Identical intervals → std=0, cv=0, stability=1.0
        let ivs = even(500.0, 16); // 120 BPM
        let s = TempoStability::from_intervals(&ivs).expect("should compute");
        assert!(
            (s.stability_score - 1.0).abs() < 1e-5,
            "stability_score should be 1.0, got {}",
            s.stability_score
        );
        assert!(!s.has_tempo_variation, "perfect metronome has no variation");
        assert!(!s.has_acceleration, "perfect metronome has no acceleration");
        assert!(s.interval_std_ms < 1e-4, "std should be ~0");
    }

    #[test]
    fn test_varying_tempo_has_variation() {
        // Wide spread of intervals → cv > 0.05
        let ivs = vec![400.0, 600.0, 400.0, 600.0, 400.0, 600.0];
        let s = TempoStability::from_intervals(&ivs).expect("should compute");
        assert!(s.has_tempo_variation, "wide spread should flag variation");
        assert!(
            s.stability_score < 1.0,
            "stability_score should be < 1.0 for varying tempo"
        );
    }

    #[test]
    fn test_acceleration_detected() {
        // Monotonically decreasing intervals (acceleration): 600→500→400→300→200
        let ivs: Vec<f32> = (0..20)
            .map(|i| 600.0 - i as f32 * 20.0)
            .filter(|&x| x > 0.0)
            .collect();
        let s = TempoStability::from_intervals(&ivs).expect("should compute");
        assert!(s.has_acceleration, "monotone decrease should flag acceleration");
    }

    #[test]
    fn test_empty_returns_none() {
        assert!(TempoStability::from_intervals(&[]).is_none());
    }

    #[test]
    fn test_single_interval_returns_none() {
        assert!(TempoStability::from_intervals(&[500.0]).is_none());
    }

    #[test]
    fn test_cv_formula_correctness() {
        // Two intervals: [100.0, 200.0]  →  mean=150, std=50, cv=1/3
        let ivs = vec![100.0_f32, 200.0];
        let s = TempoStability::from_intervals(&ivs).expect("should compute");
        let expected_mean = 150.0_f32;
        let expected_std = ((((100.0 - expected_mean).powi(2) + (200.0 - expected_mean).powi(2))
            / 2.0) as f32)
            .sqrt();
        let expected_cv = expected_std / expected_mean;
        assert!(
            (s.cv - expected_cv).abs() < 1e-4,
            "cv mismatch: got {}, expected {}",
            s.cv,
            expected_cv
        );
        assert!(
            (s.interval_std_ms - expected_std).abs() < 1e-4,
            "std mismatch: got {}, expected {}",
            s.interval_std_ms,
            expected_std
        );
    }

    #[test]
    fn test_free_function_delegates() {
        let ivs = even(500.0, 8);
        let via_method = TempoStability::from_intervals(&ivs);
        let via_fn = compute_stability(&ivs);
        assert!(via_method.is_some());
        assert!(via_fn.is_some());
        let m = via_method.expect("method result");
        let f = via_fn.expect("free fn result");
        assert!((m.stability_score - f.stability_score).abs() < 1e-6);
        assert!((m.cv - f.cv).abs() < 1e-6);
    }
}
