//! Dolby Vision tone mapping parameters.
//!
//! Provides tone curve construction, evaluation (via linear interpolation),
//! and high-level `ToneMappingParams` for HDR-to-display mapping.

#![allow(dead_code)]

// ── ToneMappingMode ───────────────────────────────────────────────────────────

/// Strategy used when mapping source HDR content to a target display.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToneMappingMode {
    /// Hard clip values above the target peak; loses highlight detail.
    Clipping,
    /// Soft roll-off near the target peak; preserves some highlight texture.
    Rolloff,
    /// S-curve statistical tone mapping; best overall tonal balance.
    STmapping,
}

impl ToneMappingMode {
    /// Returns `true` if the mode preserves some highlight detail above the
    /// target peak (i.e., not hard clipping).
    #[must_use]
    pub const fn preserves_highlights(self) -> bool {
        matches!(self, Self::Rolloff | Self::STmapping)
    }
}

// ── ToneCurvePoint ────────────────────────────────────────────────────────────

/// A single (input, output) control point on a tone curve, both in PQ [0, 1].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ToneCurvePoint {
    /// Input PQ value (must be in [0.0, 1.0]).
    pub pq_in: f32,
    /// Output PQ value (must be in [0.0, 1.0]).
    pub pq_out: f32,
}

impl ToneCurvePoint {
    /// Returns `true` when both `pq_in` and `pq_out` lie within [0.0, 1.0].
    #[must_use]
    pub fn is_valid(&self) -> bool {
        (0.0..=1.0).contains(&self.pq_in) && (0.0..=1.0).contains(&self.pq_out)
    }
}

// ── ToneCurve ─────────────────────────────────────────────────────────────────

/// A piece-wise linear tone curve defined by control points sorted by `pq_in`.
#[derive(Clone, Debug)]
pub struct ToneCurve {
    /// Control points in ascending `pq_in` order.
    pub points: Vec<ToneCurvePoint>,
}

impl ToneCurve {
    /// Create the identity tone curve (output equals input).
    #[must_use]
    pub fn new_identity() -> Self {
        Self {
            points: vec![
                ToneCurvePoint {
                    pq_in: 0.0,
                    pq_out: 0.0,
                },
                ToneCurvePoint {
                    pq_in: 1.0,
                    pq_out: 1.0,
                },
            ],
        }
    }

    /// Append a control point and keep the list sorted by `pq_in`.
    pub fn add_point(&mut self, pt: ToneCurvePoint) {
        self.points.push(pt);
        self.points.sort_by(|a, b| {
            a.pq_in
                .partial_cmp(&b.pq_in)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Evaluate the curve at `pq_in` using piece-wise linear interpolation.
    ///
    /// Values outside the defined range are clamped to the first/last point.
    #[must_use]
    pub fn evaluate(&self, pq_in: f32) -> f32 {
        if self.points.is_empty() {
            return pq_in;
        }
        // points is non-empty: the early return above guards both accesses.
        let first = &self.points[0];
        let last = &self.points[self.points.len() - 1];

        if pq_in <= first.pq_in {
            return first.pq_out;
        }
        if pq_in >= last.pq_in {
            return last.pq_out;
        }

        // Find the surrounding segment
        for window in self.points.windows(2) {
            let lo = &window[0];
            let hi = &window[1];
            if pq_in >= lo.pq_in && pq_in <= hi.pq_in {
                let t = (pq_in - lo.pq_in) / (hi.pq_in - lo.pq_in);
                return lo.pq_out + t * (hi.pq_out - lo.pq_out);
            }
        }

        // points is non-empty: empty returns pq_in at top of function.
        self.points[self.points.len() - 1].pq_out
    }

    /// Returns `true` when `pq_out` is non-decreasing along the curve.
    #[must_use]
    pub fn is_monotonic(&self) -> bool {
        self.points.windows(2).all(|w| w[1].pq_out >= w[0].pq_out)
    }
}

// ── ToneMappingParams ─────────────────────────────────────────────────────────

/// Full set of parameters describing a Dolby Vision tone mapping operation.
#[derive(Clone, Debug)]
pub struct ToneMappingParams {
    /// Algorithmic mode for the mapping.
    pub mode: ToneMappingMode,
    /// Source content peak luminance in nits.
    pub source_peak_nits: f32,
    /// Target display peak luminance in nits.
    pub target_peak_nits: f32,
    /// Tone curve applied during mapping.
    pub curve: ToneCurve,
}

impl ToneMappingParams {
    /// Ratio of target peak to source peak.
    ///
    /// Returns `1.0` when source peak is zero to avoid division by zero.
    #[must_use]
    pub fn gain_factor(&self) -> f32 {
        if self.source_peak_nits == 0.0 {
            return 1.0;
        }
        self.target_peak_nits / self.source_peak_nits
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ToneMappingMode ───────────────────────────────────────────────────────

    #[test]
    fn test_clipping_does_not_preserve_highlights() {
        assert!(!ToneMappingMode::Clipping.preserves_highlights());
    }

    #[test]
    fn test_rolloff_preserves_highlights() {
        assert!(ToneMappingMode::Rolloff.preserves_highlights());
    }

    #[test]
    fn test_stmapping_preserves_highlights() {
        assert!(ToneMappingMode::STmapping.preserves_highlights());
    }

    // ── ToneCurvePoint ────────────────────────────────────────────────────────

    #[test]
    fn test_curve_point_valid_range() {
        let p = ToneCurvePoint {
            pq_in: 0.5,
            pq_out: 0.4,
        };
        assert!(p.is_valid());
    }

    #[test]
    fn test_curve_point_invalid_pq_in_negative() {
        let p = ToneCurvePoint {
            pq_in: -0.1,
            pq_out: 0.0,
        };
        assert!(!p.is_valid());
    }

    #[test]
    fn test_curve_point_invalid_pq_out_above_one() {
        let p = ToneCurvePoint {
            pq_in: 0.5,
            pq_out: 1.1,
        };
        assert!(!p.is_valid());
    }

    #[test]
    fn test_curve_point_boundary_valid() {
        let p = ToneCurvePoint {
            pq_in: 0.0,
            pq_out: 1.0,
        };
        assert!(p.is_valid());
    }

    // ── ToneCurve ─────────────────────────────────────────────────────────────

    #[test]
    fn test_identity_curve_midpoint() {
        let c = ToneCurve::new_identity();
        let out = c.evaluate(0.5);
        assert!((out - 0.5).abs() < 1e-6, "out={out}");
    }

    #[test]
    fn test_identity_curve_is_monotonic() {
        assert!(ToneCurve::new_identity().is_monotonic());
    }

    #[test]
    fn test_curve_clamp_below() {
        let c = ToneCurve::new_identity();
        assert_eq!(c.evaluate(-1.0), 0.0);
    }

    #[test]
    fn test_curve_clamp_above() {
        let c = ToneCurve::new_identity();
        assert_eq!(c.evaluate(2.0), 1.0);
    }

    #[test]
    fn test_add_point_keeps_sorted() {
        let mut c = ToneCurve::new_identity();
        c.add_point(ToneCurvePoint {
            pq_in: 0.7,
            pq_out: 0.6,
        });
        let ins: Vec<f32> = c.points.iter().map(|p| p.pq_in).collect();
        let mut sorted = ins.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("sort_by should succeed"));
        assert_eq!(ins, sorted);
    }

    #[test]
    fn test_interpolation_midway() {
        let mut c = ToneCurve::new_identity();
        // Add a control point that compresses the upper half
        c.add_point(ToneCurvePoint {
            pq_in: 0.5,
            pq_out: 0.4,
        });
        // Between 0.0 and 0.5: out should be between 0.0 and 0.4
        let out = c.evaluate(0.25);
        assert!((out - 0.2).abs() < 1e-5, "out={out}");
    }

    #[test]
    fn test_non_monotonic_curve() {
        let c = ToneCurve {
            points: vec![
                ToneCurvePoint {
                    pq_in: 0.0,
                    pq_out: 0.5,
                },
                ToneCurvePoint {
                    pq_in: 0.5,
                    pq_out: 0.3,
                },
                ToneCurvePoint {
                    pq_in: 1.0,
                    pq_out: 1.0,
                },
            ],
        };
        assert!(!c.is_monotonic());
    }

    // ── ToneMappingParams ─────────────────────────────────────────────────────

    #[test]
    fn test_gain_factor_normal() {
        let params = ToneMappingParams {
            mode: ToneMappingMode::Rolloff,
            source_peak_nits: 4000.0,
            target_peak_nits: 1000.0,
            curve: ToneCurve::new_identity(),
        };
        assert!((params.gain_factor() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_gain_factor_zero_source_safe() {
        let params = ToneMappingParams {
            mode: ToneMappingMode::Clipping,
            source_peak_nits: 0.0,
            target_peak_nits: 1000.0,
            curve: ToneCurve::new_identity(),
        };
        assert_eq!(params.gain_factor(), 1.0);
    }

    #[test]
    fn test_gain_factor_equal_peaks() {
        let params = ToneMappingParams {
            mode: ToneMappingMode::STmapping,
            source_peak_nits: 1000.0,
            target_peak_nits: 1000.0,
            curve: ToneCurve::new_identity(),
        };
        assert!((params.gain_factor() - 1.0).abs() < 1e-6);
    }
}
