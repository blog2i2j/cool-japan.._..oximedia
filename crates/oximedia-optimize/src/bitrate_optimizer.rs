//! Bitrate optimization for video encoding in `OxiMedia`.
//!
//! Provides complexity-driven bit allocation, QP ladder management,
//! rate curves, and frame-level bitrate decision making.

#![allow(dead_code)]

/// Spatial, temporal, and noise complexity metrics for a video frame.
#[derive(Debug, Clone)]
pub struct ComplexityMetric {
    /// Spatial complexity (0.0 = flat, 1.0 = highly detailed).
    pub spatial: f32,
    /// Temporal complexity (0.0 = static, 1.0 = fast motion).
    pub temporal: f32,
    /// Noise level (0.0 = clean, 1.0 = very noisy).
    pub noise: f32,
}

impl ComplexityMetric {
    /// Returns a combined complexity score (weighted average).
    ///
    /// Weights: spatial 50 %, temporal 35 %, noise 15 %.
    #[must_use]
    pub fn combined(&self) -> f32 {
        0.50 * self.spatial + 0.35 * self.temporal + 0.15 * self.noise
    }
}

/// Per-frame bitrate allocation decision.
#[derive(Debug, Clone)]
pub struct BitrateAllocation {
    /// Zero-based frame index.
    pub frame_idx: u64,
    /// Number of bits allocated to this frame.
    pub target_bits: u32,
    /// Complexity that drove this allocation.
    pub complexity: f32,
}

impl BitrateAllocation {
    /// Returns the bits-per-pixel ratio for a frame of `width × height` pixels.
    ///
    /// Returns 0.0 if the pixel count is zero.
    #[must_use]
    pub fn bits_per_pixel(&self, width: u32, height: u32) -> f32 {
        let pixels = width as u64 * height as u64;
        if pixels == 0 {
            return 0.0;
        }
        self.target_bits as f32 / pixels as f32
    }
}

/// A QP ladder: an ordered list of quantization parameter values.
#[derive(Debug, Clone)]
pub struct QpLadder {
    /// QP values in ascending order (lower QP = higher quality, more bits).
    pub qp_values: Vec<u8>,
}

impl QpLadder {
    /// Returns the best (lowest) QP value whose bitrate estimate fits within
    /// `target_kbps`, using `max_kbps` as the denominator for linear scaling.
    ///
    /// The ladder is searched from lowest QP (highest quality) downward.
    /// If no QP fits, the highest QP value is returned.
    #[must_use]
    pub fn best_qp_for_bitrate(&self, target_kbps: u32, max_kbps: u32) -> u8 {
        if self.qp_values.is_empty() || max_kbps == 0 {
            return 51; // H.264/HEVC default max QP
        }
        // Estimate bitrate for each QP by linear inverse proportion:
        // estimated_kbps ≈ max_kbps * (1 − qp/51)
        for &qp in &self.qp_values {
            let ratio = 1.0 - (qp as f32 / 51.0);
            let estimated = (max_kbps as f32 * ratio) as u32;
            if estimated <= target_kbps {
                return qp;
            }
        }
        *self.qp_values.last().unwrap_or(&51)
    }
}

/// A piecewise-linear rate curve mapping QP → bitrate (kbps).
#[derive(Debug, Clone)]
pub struct RateCurve {
    /// (qp, kbps) pairs that define the rate curve.  Must be sorted by QP.
    pub points: Vec<(u8, u32)>,
}

impl RateCurve {
    /// Returns the linearly interpolated bitrate for a given QP.
    ///
    /// If QP is below the lowest point, the lowest bitrate is returned.
    /// If QP is above the highest point, the highest bitrate is returned.
    #[must_use]
    pub fn bitrate_for_qp(&self, qp: u8) -> u32 {
        if self.points.is_empty() {
            return 0;
        }
        if self.points.len() == 1 {
            return self.points[0].1;
        }
        // Find surrounding points (guaranteed non-empty: early-return for len() < 2 above)
        let first = self
            .points
            .first()
            .expect("points non-empty after length checks");
        let last = self
            .points
            .last()
            .expect("points non-empty after length checks");
        if qp <= first.0 {
            return first.1;
        }
        if qp >= last.0 {
            return last.1;
        }
        for i in 0..self.points.len() - 1 {
            let (q0, r0) = self.points[i];
            let (q1, r1) = self.points[i + 1];
            if qp >= q0 && qp <= q1 {
                if q1 == q0 {
                    return r0;
                }
                let t = (qp - q0) as f32 / (q1 - q0) as f32;
                return (r0 as f32 + t * (r1 as f32 - r0 as f32)) as u32;
            }
        }
        last.1
    }
}

/// High-level bitrate optimizer that combines a QP ladder with complexity history.
#[derive(Debug, Clone)]
pub struct BitrateOptimizer {
    /// QP ladder for encoding quality levels.
    pub ladder: QpLadder,
    /// Complexity metrics recorded for previously processed frames.
    pub complexity_history: Vec<ComplexityMetric>,
}

impl BitrateOptimizer {
    /// Allocates bits across `frame_count` frames using a `total_budget_kb` budget.
    ///
    /// Frames with higher complexity receive proportionally more bits.
    /// If no complexity history is available, bits are distributed evenly.
    ///
    /// # Returns
    ///
    /// A `Vec<BitrateAllocation>` with one entry per frame.
    #[must_use]
    pub fn allocate_bits(&self, frame_count: u64, total_budget_kb: u64) -> Vec<BitrateAllocation> {
        if frame_count == 0 {
            return vec![];
        }
        let total_bits = total_budget_kb * 8 * 1024;
        let complexities: Vec<f32> = if self.complexity_history.is_empty() {
            vec![1.0; frame_count as usize]
        } else {
            // Repeat/truncate history to cover all frames
            (0..frame_count as usize)
                .map(|i| self.complexity_history[i % self.complexity_history.len()].combined())
                .collect()
        };
        let sum: f32 = complexities.iter().sum::<f32>().max(f32::EPSILON);
        complexities
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let fraction = c / sum;
                BitrateAllocation {
                    frame_idx: i as u64,
                    target_bits: (total_bits as f32 * fraction) as u32,
                    complexity: c,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complexity_combined_flat() {
        let c = ComplexityMetric {
            spatial: 0.0,
            temporal: 0.0,
            noise: 0.0,
        };
        assert!((c.combined() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_complexity_combined_all_ones() {
        let c = ComplexityMetric {
            spatial: 1.0,
            temporal: 1.0,
            noise: 1.0,
        };
        assert!((c.combined() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_complexity_combined_weights() {
        let c = ComplexityMetric {
            spatial: 1.0,
            temporal: 0.0,
            noise: 0.0,
        };
        assert!((c.combined() - 0.50).abs() < 1e-5);
    }

    #[test]
    fn test_bitrate_allocation_bits_per_pixel() {
        let a = BitrateAllocation {
            frame_idx: 0,
            target_bits: 4_000,
            complexity: 0.5,
        };
        let bpp = a.bits_per_pixel(40, 25); // 1000 pixels
        assert!((bpp - 4.0).abs() < 1e-4);
    }

    #[test]
    fn test_bitrate_allocation_bpp_zero_pixels() {
        let a = BitrateAllocation {
            frame_idx: 0,
            target_bits: 1000,
            complexity: 0.5,
        };
        assert_eq!(a.bits_per_pixel(0, 100), 0.0);
    }

    #[test]
    fn test_qp_ladder_best_qp_high_target() {
        let ladder = QpLadder {
            qp_values: vec![18, 24, 30, 36, 42],
        };
        // With max_kbps = 10000 and target = 10000, lowest QP should fit
        let qp = ladder.best_qp_for_bitrate(10_000, 10_000);
        assert_eq!(qp, 18, "got {qp}");
    }

    #[test]
    fn test_qp_ladder_best_qp_low_target() {
        let ladder = QpLadder {
            qp_values: vec![18, 24, 30, 36, 42],
        };
        // Very low target → should return highest QP
        let qp = ladder.best_qp_for_bitrate(1, 10_000);
        assert_eq!(qp, 42);
    }

    #[test]
    fn test_qp_ladder_empty() {
        let ladder = QpLadder { qp_values: vec![] };
        assert_eq!(ladder.best_qp_for_bitrate(5000, 10_000), 51);
    }

    #[test]
    fn test_rate_curve_interpolation() {
        let curve = RateCurve {
            points: vec![(20, 8000), (30, 4000), (40, 2000)],
        };
        // QP 25 = midpoint between 20 and 30 → ~6000
        let r = curve.bitrate_for_qp(25);
        assert!((r as i64 - 6000).abs() < 50, "got {r}");
    }

    #[test]
    fn test_rate_curve_clamp_below() {
        let curve = RateCurve {
            points: vec![(20, 8000), (40, 2000)],
        };
        assert_eq!(curve.bitrate_for_qp(10), 8000);
    }

    #[test]
    fn test_rate_curve_clamp_above() {
        let curve = RateCurve {
            points: vec![(20, 8000), (40, 2000)],
        };
        assert_eq!(curve.bitrate_for_qp(50), 2000);
    }

    #[test]
    fn test_rate_curve_empty() {
        let curve = RateCurve { points: vec![] };
        assert_eq!(curve.bitrate_for_qp(28), 0);
    }

    #[test]
    fn test_bitrate_optimizer_allocate_even() {
        let opt = BitrateOptimizer {
            ladder: QpLadder {
                qp_values: vec![24, 28, 32],
            },
            complexity_history: vec![],
        };
        let allocs = opt.allocate_bits(4, 100);
        assert_eq!(allocs.len(), 4);
        // All complexities equal → all allocations equal
        let bits: Vec<u32> = allocs.iter().map(|a| a.target_bits).collect();
        assert_eq!(bits[0], bits[1]);
        assert_eq!(bits[1], bits[2]);
    }

    #[test]
    fn test_bitrate_optimizer_allocate_weighted() {
        let opt = BitrateOptimizer {
            ladder: QpLadder {
                qp_values: vec![24, 28, 32],
            },
            complexity_history: vec![
                ComplexityMetric {
                    spatial: 0.2,
                    temporal: 0.1,
                    noise: 0.0,
                },
                ComplexityMetric {
                    spatial: 0.8,
                    temporal: 0.9,
                    noise: 0.5,
                },
            ],
        };
        let allocs = opt.allocate_bits(2, 100);
        // Second frame is more complex → more bits
        assert!(allocs[1].target_bits > allocs[0].target_bits);
    }

    #[test]
    fn test_bitrate_optimizer_empty_frames() {
        let opt = BitrateOptimizer {
            ladder: QpLadder {
                qp_values: vec![28],
            },
            complexity_history: vec![],
        };
        assert!(opt.allocate_bits(0, 1000).is_empty());
    }

    #[test]
    fn test_bitrate_optimizer_total_budget_consumed() {
        let opt = BitrateOptimizer {
            ladder: QpLadder {
                qp_values: vec![24, 28, 32],
            },
            complexity_history: vec![],
        };
        let budget_kb = 50u64;
        let allocs = opt.allocate_bits(10, budget_kb);
        let total: u64 = allocs.iter().map(|a| a.target_bits as u64).sum();
        let expected = budget_kb * 8 * 1024;
        // Allow for rounding errors
        assert!(
            (total as i64 - expected as i64).unsigned_abs() < 10,
            "total={total}, expected={expected}"
        );
    }
}
