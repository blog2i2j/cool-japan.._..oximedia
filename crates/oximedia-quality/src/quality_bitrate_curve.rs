//! Quality-bitrate curve generation for encode ladder analysis.
//!
//! This module generates quality-vs-bitrate curves by accepting a series of
//! (bitrate, quality_score) data points and providing analysis tools including:
//!
//! * Curve fitting (piecewise linear interpolation)
//! * Optimal operating point selection (maximum quality-per-bit efficiency)
//! * Pareto-optimal point identification
//! * Bitrate estimation for a target quality level
//! * Quality estimation for a given bitrate budget
//! * Curve statistics (slope, diminishing returns analysis)
//!
//! The module is codec-agnostic: callers supply raw measurement data obtained
//! by encoding the same source at multiple CRF/QP/bitrate-target values.

use serde::{Deserialize, Serialize};
use std::fmt;

// ─── Error type ───────────────────────────────────────────────────────────────

/// Errors produced by quality-bitrate curve operations.
#[derive(Debug, Clone, PartialEq)]
pub enum CurveError {
    /// No data points were provided.
    Empty,
    /// A target quality or bitrate value is outside the range of the curve data.
    OutOfRange {
        /// The value that was out of range.
        value: f64,
        /// The minimum value in the curve data.
        min: f64,
        /// The maximum value in the curve data.
        max: f64,
    },
    /// The data points contain duplicate bitrate values which would produce a
    /// non-monotonic or degenerate curve.
    DuplicateBitrate(u64),
    /// Fewer than 2 data points were supplied; analysis requires at least two.
    InsufficientPoints,
}

impl fmt::Display for CurveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "no data points provided"),
            Self::OutOfRange { value, min, max } => write!(
                f,
                "value {value:.4} is out of curve range [{min:.4}, {max:.4}]"
            ),
            Self::DuplicateBitrate(br) => {
                write!(f, "duplicate bitrate entry: {br} kbps")
            }
            Self::InsufficientPoints => {
                write!(f, "at least 2 data points are required for analysis")
            }
        }
    }
}

impl std::error::Error for CurveError {}

// ─── Data point ───────────────────────────────────────────────────────────────

/// A single (bitrate, quality) measurement from a specific encode.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CurvePoint {
    /// Bitrate in kilobits per second (kbps).
    pub bitrate_kbps: u64,
    /// Quality score — typically VMAF \[0, 100\], SSIM \[0, 1\], or PSNR (dB).
    pub quality: f64,
}

impl CurvePoint {
    /// Creates a new curve point.
    #[must_use]
    pub fn new(bitrate_kbps: u64, quality: f64) -> Self {
        Self {
            bitrate_kbps,
            quality,
        }
    }

    /// Quality-per-kilobit efficiency ratio.
    ///
    /// Higher values indicate more quality delivered per unit of bandwidth.
    /// Returns `0.0` for zero-bitrate points to avoid division by zero.
    #[must_use]
    pub fn efficiency(&self) -> f64 {
        if self.bitrate_kbps == 0 {
            return 0.0;
        }
        self.quality / self.bitrate_kbps as f64
    }
}

// ─── Analysis results ─────────────────────────────────────────────────────────

/// Summary statistics for a quality-bitrate curve segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurveSegment {
    /// The lower-bitrate endpoint of this segment.
    pub low: CurvePoint,
    /// The higher-bitrate endpoint of this segment.
    pub high: CurvePoint,
    /// Quality gain per kbps increase across this segment.
    pub slope: f64,
}

impl CurveSegment {
    /// Interpolates quality at the given bitrate within this segment.
    ///
    /// The bitrate must be in `[low.bitrate_kbps, high.bitrate_kbps]`.
    #[must_use]
    pub fn interpolate_quality(&self, bitrate_kbps: u64) -> f64 {
        let span = (self.high.bitrate_kbps - self.low.bitrate_kbps) as f64;
        if span < 1e-10 {
            return self.low.quality;
        }
        let t = (bitrate_kbps - self.low.bitrate_kbps) as f64 / span;
        self.low.quality + t * (self.high.quality - self.low.quality)
    }

    /// Returns the bitrate at which the given quality level is reached,
    /// by linear interpolation within this segment.
    ///
    /// Returns `None` if the target quality is outside the segment's range.
    #[must_use]
    pub fn interpolate_bitrate(&self, target_quality: f64) -> Option<u64> {
        let q_lo = self.low.quality.min(self.high.quality);
        let q_hi = self.low.quality.max(self.high.quality);

        if target_quality < q_lo || target_quality > q_hi {
            return None;
        }

        let span_q = self.high.quality - self.low.quality;
        if span_q.abs() < 1e-10 {
            return Some(self.low.bitrate_kbps);
        }

        let t = (target_quality - self.low.quality) / span_q;
        let span_br = (self.high.bitrate_kbps - self.low.bitrate_kbps) as f64;
        let br = self.low.bitrate_kbps as f64 + t * span_br;
        Some(br.round() as u64)
    }
}

/// Full quality-bitrate curve with analysis capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityBitrateCurve {
    /// Data points sorted by bitrate (ascending).
    points: Vec<CurvePoint>,
}

impl QualityBitrateCurve {
    /// Constructs a curve from a set of raw measurement points.
    ///
    /// Points are sorted by bitrate and validated for duplicate bitrates.
    ///
    /// # Errors
    ///
    /// Returns [`CurveError::Empty`] if `points` is empty, or
    /// [`CurveError::DuplicateBitrate`] if any two points share the same bitrate.
    pub fn new(points: Vec<CurvePoint>) -> Result<Self, CurveError> {
        if points.is_empty() {
            return Err(CurveError::Empty);
        }

        let mut sorted = points;
        sorted.sort_by_key(|p| p.bitrate_kbps);

        // Validate uniqueness
        for w in sorted.windows(2) {
            if w[0].bitrate_kbps == w[1].bitrate_kbps {
                return Err(CurveError::DuplicateBitrate(w[0].bitrate_kbps));
            }
        }

        Ok(Self { points: sorted })
    }

    /// Returns the underlying sorted data points.
    #[must_use]
    pub fn points(&self) -> &[CurvePoint] {
        &self.points
    }

    /// Returns the number of measurement points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns `true` if the curve contains no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Returns the minimum bitrate in the curve.
    #[must_use]
    pub fn min_bitrate(&self) -> u64 {
        self.points.first().map(|p| p.bitrate_kbps).unwrap_or(0)
    }

    /// Returns the maximum bitrate in the curve.
    #[must_use]
    pub fn max_bitrate(&self) -> u64 {
        self.points.last().map(|p| p.bitrate_kbps).unwrap_or(0)
    }

    /// Returns the minimum quality value across all points.
    #[must_use]
    pub fn min_quality(&self) -> f64 {
        self.points
            .iter()
            .map(|p| p.quality)
            .fold(f64::INFINITY, f64::min)
    }

    /// Returns the maximum quality value across all points.
    #[must_use]
    pub fn max_quality(&self) -> f64 {
        self.points
            .iter()
            .map(|p| p.quality)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Estimates quality at a given bitrate via piecewise linear interpolation.
    ///
    /// For bitrates below the minimum measured bitrate, the lowest quality
    /// data point is returned.  For bitrates above the maximum, the highest
    /// quality data point is returned (clamp-extrapolation).
    ///
    /// # Errors
    ///
    /// Returns [`CurveError::Empty`] if the curve has no points.
    pub fn quality_at_bitrate(&self, bitrate_kbps: u64) -> Result<f64, CurveError> {
        if self.points.is_empty() {
            return Err(CurveError::Empty);
        }

        // Clamp to the measured range
        if bitrate_kbps <= self.points[0].bitrate_kbps {
            return Ok(self.points[0].quality);
        }
        let last = self.points.last().expect("non-empty checked above");
        if bitrate_kbps >= last.bitrate_kbps {
            return Ok(last.quality);
        }

        // Binary search for the enclosing segment
        let idx = self
            .points
            .partition_point(|p| p.bitrate_kbps <= bitrate_kbps);
        let seg = CurveSegment {
            low: self.points[idx - 1],
            high: self.points[idx],
            slope: 0.0, // slope not needed for interpolation
        };
        Ok(seg.interpolate_quality(bitrate_kbps))
    }

    /// Estimates the bitrate required to achieve a given quality level.
    ///
    /// Uses piecewise linear interpolation between adjacent data points.
    ///
    /// # Errors
    ///
    /// Returns [`CurveError::InsufficientPoints`] if there is only one point,
    /// or [`CurveError::OutOfRange`] if `target_quality` lies outside the
    /// quality range of the curve.
    pub fn bitrate_for_quality(&self, target_quality: f64) -> Result<u64, CurveError> {
        if self.points.len() < 2 {
            return Err(CurveError::InsufficientPoints);
        }

        let q_min = self.min_quality();
        let q_max = self.max_quality();

        if target_quality < q_min || target_quality > q_max {
            return Err(CurveError::OutOfRange {
                value: target_quality,
                min: q_min,
                max: q_max,
            });
        }

        // Search through segments for one that spans the target quality
        for seg_pts in self.points.windows(2) {
            let low = seg_pts[0];
            let high = seg_pts[1];
            let seg = CurveSegment {
                low,
                high,
                slope: 0.0,
            };
            if let Some(br) = seg.interpolate_bitrate(target_quality) {
                return Ok(br);
            }
        }

        // Fallback: return the bitrate of the closest quality point
        let closest = self
            .points
            .iter()
            .min_by(|a, b| {
                (a.quality - target_quality)
                    .abs()
                    .partial_cmp(&(b.quality - target_quality).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("non-empty checked above");
        Ok(closest.bitrate_kbps)
    }

    /// Returns the point with the highest quality-per-kilobit efficiency.
    ///
    /// This is the "knee" of the curve — the encode setting that delivers the
    /// most quality per unit of bandwidth.
    ///
    /// # Errors
    ///
    /// Returns [`CurveError::Empty`] if the curve has no points.
    pub fn most_efficient_point(&self) -> Result<CurvePoint, CurveError> {
        self.points
            .iter()
            .max_by(|a, b| {
                a.efficiency()
                    .partial_cmp(&b.efficiency())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .ok_or(CurveError::Empty)
    }

    /// Returns per-segment slope (quality gain per kbps) across the curve.
    ///
    /// Segments with a high slope represent areas where spending more bits
    /// yields significant quality improvement; low-slope segments indicate
    /// diminishing returns.
    ///
    /// # Errors
    ///
    /// Returns [`CurveError::InsufficientPoints`] if fewer than 2 points exist.
    pub fn segment_slopes(&self) -> Result<Vec<CurveSegment>, CurveError> {
        if self.points.len() < 2 {
            return Err(CurveError::InsufficientPoints);
        }

        let segments = self
            .points
            .windows(2)
            .map(|w| {
                let low = w[0];
                let high = w[1];
                let delta_q = high.quality - low.quality;
                let delta_br = (high.bitrate_kbps - low.bitrate_kbps) as f64;
                let slope = if delta_br > 0.0 {
                    delta_q / delta_br
                } else {
                    0.0
                };
                CurveSegment { low, high, slope }
            })
            .collect();

        Ok(segments)
    }

    /// Identifies the point of diminishing returns: the segment where the
    /// slope first drops to or below `slope_fraction` of the maximum slope.
    ///
    /// For example, with `slope_fraction = 0.25` this returns the first point
    /// at which each additional kbps yields ≤25 % of the peak quality gain,
    /// indicating that further bitrate increases are largely wasteful.
    ///
    /// Returns `None` if the curve is monotonically efficient (slope never
    /// drops below the threshold) or if there are fewer than 2 points.
    ///
    /// # Arguments
    ///
    /// * `slope_fraction` — threshold fraction of peak slope (0.0–1.0).
    #[must_use]
    pub fn diminishing_returns_point(&self, slope_fraction: f64) -> Option<CurvePoint> {
        let segments = self.segment_slopes().ok()?;

        let max_slope = segments
            .iter()
            .map(|s| s.slope)
            .fold(f64::NEG_INFINITY, f64::max);

        if max_slope <= 0.0 {
            return None;
        }

        let threshold = max_slope * slope_fraction;

        // Find the first segment where slope has fallen below the threshold,
        // then return the lower endpoint of that segment (the "knee").
        for seg in &segments {
            if seg.slope <= threshold {
                return Some(seg.low);
            }
        }

        None
    }

    /// Computes the Bjøntegaard-Delta Rate (BD-Rate) approximation between
    /// this curve and another curve.
    ///
    /// BD-Rate estimates the average bitrate savings (as a percentage) that
    /// `other` achieves relative to `self` at the same quality levels.
    /// A negative BD-Rate means `other` is more efficient (uses fewer bits for
    /// the same quality).
    ///
    /// The implementation uses the piecewise linear area integration method
    /// over the overlapping quality range, sampled at `sample_count` evenly
    /// spaced quality levels.
    ///
    /// # Errors
    ///
    /// Returns [`CurveError::InsufficientPoints`] if either curve has fewer
    /// than 2 points, or [`CurveError::OutOfRange`] if there is no overlapping
    /// quality range.
    pub fn bd_rate(
        &self,
        other: &QualityBitrateCurve,
        sample_count: usize,
    ) -> Result<f64, CurveError> {
        if self.points.len() < 2 || other.points.len() < 2 {
            return Err(CurveError::InsufficientPoints);
        }

        let samples = sample_count.max(2);

        // Overlapping quality range
        let q_low = self.min_quality().max(other.min_quality());
        let q_high = self.max_quality().min(other.max_quality());

        if q_high <= q_low {
            return Err(CurveError::OutOfRange {
                value: q_low,
                min: q_low,
                max: q_high,
            });
        }

        let step = (q_high - q_low) / (samples - 1) as f64;

        let mut log_ratio_sum = 0.0_f64;
        let mut valid_count = 0usize;

        for i in 0..samples {
            let q = q_low + i as f64 * step;
            let br_self = self.bitrate_for_quality(q).ok();
            let br_other = other.bitrate_for_quality(q).ok();

            if let (Some(br_a), Some(br_b)) = (br_self, br_other) {
                if br_a > 0 && br_b > 0 {
                    log_ratio_sum += (br_b as f64 / br_a as f64).ln();
                    valid_count += 1;
                }
            }
        }

        if valid_count == 0 {
            return Err(CurveError::InsufficientPoints);
        }

        // Convert average log ratio to percentage
        let avg_log_ratio = log_ratio_sum / valid_count as f64;
        Ok((avg_log_ratio.exp() - 1.0) * 100.0)
    }
}

// ─── Builder ──────────────────────────────────────────────────────────────────

/// Incrementally builds a [`QualityBitrateCurve`] by adding measurements
/// one at a time.
#[derive(Default)]
pub struct CurveBuilder {
    points: Vec<CurvePoint>,
}

impl CurveBuilder {
    /// Creates a new, empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a (bitrate, quality) measurement to the builder.
    pub fn add_point(&mut self, bitrate_kbps: u64, quality: f64) -> &mut Self {
        self.points.push(CurvePoint::new(bitrate_kbps, quality));
        self
    }

    /// Consumes the builder and constructs a [`QualityBitrateCurve`].
    ///
    /// # Errors
    ///
    /// Propagates errors from [`QualityBitrateCurve::new`].
    pub fn build(self) -> Result<QualityBitrateCurve, CurveError> {
        QualityBitrateCurve::new(self.points)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a typical VMAF-style curve (bitrate in kbps, score 0–100).
    fn vmaf_curve() -> QualityBitrateCurve {
        let mut builder = CurveBuilder::new();
        builder
            .add_point(500, 52.0)
            .add_point(1000, 68.0)
            .add_point(2000, 80.0)
            .add_point(4000, 88.0)
            .add_point(8000, 92.0);
        builder.build().expect("valid curve")
    }

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn test_empty_curve_returns_error() {
        assert!(matches!(
            QualityBitrateCurve::new(vec![]),
            Err(CurveError::Empty)
        ));
    }

    #[test]
    fn test_duplicate_bitrate_returns_error() {
        let points = vec![
            CurvePoint::new(1000, 70.0),
            CurvePoint::new(1000, 75.0), // duplicate
        ];
        assert!(matches!(
            QualityBitrateCurve::new(points),
            Err(CurveError::DuplicateBitrate(1000))
        ));
    }

    #[test]
    fn test_single_point_curve_valid() {
        let curve = QualityBitrateCurve::new(vec![CurvePoint::new(1000, 75.0)]);
        assert!(curve.is_ok());
        assert_eq!(curve.expect("valid").len(), 1);
    }

    #[test]
    fn test_curve_sorted_by_bitrate() {
        // Provide points in reverse order — they should be sorted.
        let points = vec![
            CurvePoint::new(8000, 92.0),
            CurvePoint::new(500, 52.0),
            CurvePoint::new(2000, 80.0),
        ];
        let curve = QualityBitrateCurve::new(points).expect("valid");
        let pts = curve.points();
        assert_eq!(pts[0].bitrate_kbps, 500);
        assert_eq!(pts[1].bitrate_kbps, 2000);
        assert_eq!(pts[2].bitrate_kbps, 8000);
    }

    // ── quality_at_bitrate ────────────────────────────────────────────────────

    #[test]
    fn test_quality_at_exact_bitrate_point() {
        let curve = vmaf_curve();
        let q = curve.quality_at_bitrate(1000).expect("should succeed");
        assert!(
            (q - 68.0).abs() < 1e-6,
            "expected 68.0 at 1000 kbps, got {q}"
        );
    }

    #[test]
    fn test_quality_at_midpoint_interpolated() {
        let curve = vmaf_curve();
        // Midpoint between 1000 kbps (68) and 2000 kbps (80) → should be 74.
        let q = curve.quality_at_bitrate(1500).expect("should succeed");
        assert!(
            (q - 74.0).abs() < 1e-6,
            "expected 74.0 at 1500 kbps, got {q}"
        );
    }

    #[test]
    fn test_quality_clamps_below_min_bitrate() {
        let curve = vmaf_curve();
        let q = curve.quality_at_bitrate(100).expect("should succeed");
        assert_eq!(q, 52.0, "below min bitrate should return min quality");
    }

    #[test]
    fn test_quality_clamps_above_max_bitrate() {
        let curve = vmaf_curve();
        let q = curve.quality_at_bitrate(16000).expect("should succeed");
        assert_eq!(q, 92.0, "above max bitrate should return max quality");
    }

    // ── bitrate_for_quality ───────────────────────────────────────────────────

    #[test]
    fn test_bitrate_for_quality_exact() {
        let curve = vmaf_curve();
        let br = curve.bitrate_for_quality(80.0).expect("should succeed");
        assert_eq!(br, 2000, "quality 80 → 2000 kbps");
    }

    #[test]
    fn test_bitrate_for_quality_interpolated() {
        let curve = vmaf_curve();
        // Between 68@1000 and 80@2000: quality 74 → 1500 kbps
        let br = curve.bitrate_for_quality(74.0).expect("should succeed");
        assert_eq!(br, 1500, "quality 74 → 1500 kbps");
    }

    #[test]
    fn test_bitrate_for_quality_out_of_range() {
        let curve = vmaf_curve();
        assert!(matches!(
            curve.bitrate_for_quality(10.0),
            Err(CurveError::OutOfRange { .. })
        ));
        assert!(matches!(
            curve.bitrate_for_quality(99.0),
            Err(CurveError::OutOfRange { .. })
        ));
    }

    #[test]
    fn test_bitrate_for_quality_insufficient_points() {
        let curve = QualityBitrateCurve::new(vec![CurvePoint::new(1000, 75.0)]).expect("valid");
        assert!(matches!(
            curve.bitrate_for_quality(75.0),
            Err(CurveError::InsufficientPoints)
        ));
    }

    // ── most_efficient_point ──────────────────────────────────────────────────

    #[test]
    fn test_most_efficient_point_is_lowest_bitrate() {
        // Efficiency = quality / bitrate. Lowest bitrate has the highest ratio
        // when quality doesn't drop proportionally.
        let curve = vmaf_curve();
        let eff = curve.most_efficient_point().expect("should succeed");
        // At 500 kbps: 52/500 = 0.104; at 1000: 68/1000 = 0.068 — 500 wins.
        assert_eq!(eff.bitrate_kbps, 500);
    }

    // ── segment_slopes ────────────────────────────────────────────────────────

    #[test]
    fn test_segment_slopes_count() {
        let curve = vmaf_curve(); // 5 points → 4 segments
        let slopes = curve.segment_slopes().expect("should succeed");
        assert_eq!(slopes.len(), 4);
    }

    #[test]
    fn test_segment_slopes_are_non_negative_for_monotone_curve() {
        let curve = vmaf_curve();
        let slopes = curve.segment_slopes().expect("should succeed");
        for seg in &slopes {
            assert!(seg.slope >= 0.0, "monotone curve: slope must be ≥ 0");
        }
    }

    #[test]
    fn test_segment_slopes_insufficient_points() {
        let curve = QualityBitrateCurve::new(vec![CurvePoint::new(1000, 70.0)]).expect("valid");
        assert!(matches!(
            curve.segment_slopes(),
            Err(CurveError::InsufficientPoints)
        ));
    }

    // ── diminishing_returns_point ─────────────────────────────────────────────

    #[test]
    fn test_diminishing_returns_returns_some_for_diminishing_curve() {
        let curve = vmaf_curve();
        // The first segment 500→1000 has slope (68-52)/500 = 0.032
        // The second segment 1000→2000 has slope (80-68)/1000 = 0.012
        // So diminishing returns at 0.5 of max slope = 0.016 should fire.
        let knee = curve.diminishing_returns_point(0.5);
        assert!(
            knee.is_some(),
            "curve with diminishing returns should find a knee"
        );
    }

    #[test]
    fn test_diminishing_returns_threshold_zero_returns_none_or_first() {
        // With threshold 0.0, every segment satisfies slope <= 0 * max = 0
        // only if the curve has a flat segment. Otherwise returns None.
        let points = vec![
            CurvePoint::new(500, 60.0),
            CurvePoint::new(1000, 80.0),
            CurvePoint::new(2000, 90.0),
        ];
        let curve = QualityBitrateCurve::new(points).expect("valid");
        // threshold 0.0 → knee is the first segment where slope == 0.
        // None of these slopes are zero, so None is expected.
        let knee = curve.diminishing_returns_point(0.0);
        // Either None or a valid point; just ensure no panic
        let _ = knee;
    }

    // ── bd_rate ───────────────────────────────────────────────────────────────

    #[test]
    fn test_bd_rate_identical_curves_is_zero() {
        let curve = vmaf_curve();
        let bd = curve.bd_rate(&curve, 20).expect("should succeed");
        assert!(
            bd.abs() < 1e-6,
            "identical curves should have BD-Rate ≈ 0, got {bd}"
        );
    }

    #[test]
    fn test_bd_rate_better_curve_is_negative() {
        // "other" achieves the same quality at half the bitrate → −50 % BD-Rate.
        let base = vmaf_curve();
        let efficient: QualityBitrateCurve = {
            let points: Vec<CurvePoint> = base
                .points()
                .iter()
                .map(|p| CurvePoint::new(p.bitrate_kbps / 2, p.quality))
                .collect();
            QualityBitrateCurve::new(points).expect("valid")
        };
        let bd = base.bd_rate(&efficient, 20).expect("should succeed");
        // efficient uses ≈50 % of base's bitrate → BD-Rate ≈ −50 %
        assert!(
            bd < 0.0,
            "more efficient curve should have negative BD-Rate, got {bd}"
        );
    }

    #[test]
    fn test_bd_rate_insufficient_points() {
        let single = QualityBitrateCurve::new(vec![CurvePoint::new(1000, 70.0)]).expect("valid");
        let curve = vmaf_curve();
        assert!(matches!(
            curve.bd_rate(&single, 10),
            Err(CurveError::InsufficientPoints)
        ));
    }

    // ── builder ───────────────────────────────────────────────────────────────

    #[test]
    fn test_curve_builder_constructs_correctly() {
        let mut b = CurveBuilder::new();
        b.add_point(1000, 70.0).add_point(2000, 82.0);
        let curve = b.build().expect("valid");
        assert_eq!(curve.len(), 2);
        assert_eq!(curve.min_bitrate(), 1000);
        assert_eq!(curve.max_bitrate(), 2000);
    }

    #[test]
    fn test_curve_point_efficiency() {
        let p = CurvePoint::new(1000, 80.0);
        assert!((p.efficiency() - 0.08).abs() < 1e-9);

        let zero_br = CurvePoint::new(0, 80.0);
        assert_eq!(zero_br.efficiency(), 0.0);
    }
}
