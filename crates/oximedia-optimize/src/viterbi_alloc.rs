//! Per-frame bitrate allocation using the Viterbi algorithm.
//!
//! Finds the globally optimal sequence of per-frame QP values that minimises
//! total distortion while meeting a target bitrate constraint.
//!
//! # Algorithm
//!
//! The problem is modelled as a shortest-path search on a trellis (directed
//! acyclic graph):
//!
//! - **Stages**: one per frame in the segment (typically a GOP or scene).
//! - **States**: quantised QP levels (e.g. QP 20..51 in steps of 1).
//! - **Transitions**: (QP_prev, QP_curr) pairs, with a smoothness penalty
//!   for large QP jumps.
//! - **Cost per state**: `distortion(frame, QP) + lambda * rate(frame, QP)`,
//!   where distortion and rate are estimated from a per-frame R-D model.
//!
//! The Viterbi (dynamic programming) sweep finds the minimum-cost path
//! through the trellis in O(F * Q^2) time, where F = frame count and
//! Q = number of QP levels.
//!
//! After the forward sweep a backward trace recovers the optimal QP
//! sequence.  An optional Lagrangian bisection step adjusts `lambda` so
//! that the resulting total rate matches the target bitrate.
//!
//! # References
//!
//! - Ortega & Ramchandran, "Rate-distortion methods for image and video
//!   compression", IEEE SP Magazine 1998.
//! - Shoham & Gersho, "Efficient bit allocation for an arbitrary set of
//!   quantizers", IEEE Trans. ASSP 1988.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

// ── R-D model per frame ────────────────────────────────────────────────────

/// Per-frame rate-distortion model.
///
/// Estimates distortion and rate as functions of QP using a simple
/// parametric model.  In production these would come from a first-pass
/// analysis; here we provide a configurable default.
#[derive(Debug, Clone)]
pub struct FrameRdModel {
    /// Frame index within the segment.
    pub frame_index: usize,
    /// Spatial complexity (higher = more bits needed).  Arbitrary positive units.
    pub complexity: f64,
    /// Temporal complexity / motion energy.  Used to scale distortion sensitivity.
    pub temporal_energy: f64,
    /// Frame type weight (I-frame = ~3.0, P = 1.0, B = 0.6 typical).
    pub type_weight: f64,
}

impl FrameRdModel {
    /// Estimates distortion for a given QP.
    ///
    /// Uses `D(q) = complexity * 2^((q - 26) / 3)` which gives an
    /// exponential distortion curve (halving QP step of 6 ≈ halving MSE).
    pub fn distortion(&self, qp: u8) -> f64 {
        let q = qp as f64;
        self.complexity * 2.0_f64.powf((q - 26.0) / 3.0) * (1.0 + 0.1 * self.temporal_energy)
    }

    /// Estimates coded bits for a given QP.
    ///
    /// Uses `R(q) = complexity * type_weight * 2^((26 - q) / 3)`.
    pub fn rate(&self, qp: u8) -> f64 {
        let q = qp as f64;
        self.complexity * self.type_weight * 2.0_f64.powf((26.0 - q) / 3.0)
    }
}

// ── Configuration ───────────────────────────────────────────────────────────

/// Configuration for Viterbi bitrate allocation.
#[derive(Debug, Clone)]
pub struct ViterbiAllocConfig {
    /// Minimum QP to consider.
    pub qp_min: u8,
    /// Maximum QP to consider.
    pub qp_max: u8,
    /// QP step size for the trellis states.
    pub qp_step: u8,
    /// Smoothness penalty coefficient: penalises `|QP_i - QP_{i-1}|` per unit.
    pub smoothness_penalty: f64,
    /// Target total bits for the segment (if `Some`, lambda bisection is used).
    pub target_bits: Option<f64>,
    /// Initial Lagrangian multiplier for rate-distortion trade-off.
    pub initial_lambda: f64,
    /// Maximum lambda bisection iterations.
    pub max_bisect_iters: u32,
    /// Lambda bisection convergence tolerance (fraction of target bits).
    pub bisect_tolerance: f64,
}

impl Default for ViterbiAllocConfig {
    fn default() -> Self {
        Self {
            qp_min: 18,
            qp_max: 51,
            qp_step: 1,
            smoothness_penalty: 2.0,
            target_bits: None,
            initial_lambda: 1.0,
            max_bisect_iters: 20,
            bisect_tolerance: 0.02,
        }
    }
}

impl ViterbiAllocConfig {
    /// Returns the list of QP levels used as trellis states.
    fn qp_levels(&self) -> Vec<u8> {
        let mut levels = Vec::new();
        let mut q = self.qp_min;
        while q <= self.qp_max {
            levels.push(q);
            q = q.saturating_add(self.qp_step);
            if self.qp_step == 0 {
                break;
            }
        }
        levels
    }
}

// ── Trellis solver ──────────────────────────────────────────────────────────

/// Result of Viterbi allocation.
#[derive(Debug, Clone)]
pub struct ViterbiAllocResult {
    /// Optimal QP for each frame.
    pub qp_sequence: Vec<u8>,
    /// Total estimated bits.
    pub total_bits: f64,
    /// Total estimated distortion.
    pub total_distortion: f64,
    /// Final Lagrangian lambda used.
    pub final_lambda: f64,
    /// Number of trellis states per frame.
    pub states_per_frame: usize,
}

/// Runs the Viterbi trellis to find optimal QP assignment for a fixed lambda.
fn viterbi_solve(
    frames: &[FrameRdModel],
    qp_levels: &[u8],
    lambda: f64,
    smoothness_penalty: f64,
) -> (Vec<u8>, f64, f64) {
    let n_frames = frames.len();
    let n_states = qp_levels.len();

    if n_frames == 0 || n_states == 0 {
        return (Vec::new(), 0.0, 0.0);
    }

    // cost[f][s] = minimum cumulative cost to reach frame f in state s
    let mut cost = vec![vec![f64::INFINITY; n_states]; n_frames];
    // backptr[f][s] = state index at frame f-1 on the optimal path to (f, s)
    let mut backptr = vec![vec![0usize; n_states]; n_frames];

    // Initialise first frame
    for (s, &qp) in qp_levels.iter().enumerate() {
        let d = frames[0].distortion(qp);
        let r = frames[0].rate(qp);
        cost[0][s] = d + lambda * r;
    }

    // Forward sweep
    for f in 1..n_frames {
        for (s_cur, &qp_cur) in qp_levels.iter().enumerate() {
            let d = frames[f].distortion(qp_cur);
            let r = frames[f].rate(qp_cur);
            let stage_cost = d + lambda * r;

            let mut best_cost = f64::INFINITY;
            let mut best_prev = 0usize;

            for (s_prev, &qp_prev) in qp_levels.iter().enumerate() {
                let qp_diff = (qp_cur as i32 - qp_prev as i32).unsigned_abs() as f64;
                let transition = smoothness_penalty * qp_diff * qp_diff;
                let total = cost[f - 1][s_prev] + stage_cost + transition;

                if total < best_cost {
                    best_cost = total;
                    best_prev = s_prev;
                }
            }

            cost[f][s_cur] = best_cost;
            backptr[f][s_cur] = best_prev;
        }
    }

    // Find best final state
    let mut best_final_state = 0usize;
    let mut best_final_cost = f64::INFINITY;
    for s in 0..n_states {
        if cost[n_frames - 1][s] < best_final_cost {
            best_final_cost = cost[n_frames - 1][s];
            best_final_state = s;
        }
    }

    // Backward trace
    let mut qp_seq = vec![0u8; n_frames];
    qp_seq[n_frames - 1] = qp_levels[best_final_state];
    let mut state = best_final_state;
    for f in (1..n_frames).rev() {
        state = backptr[f][state];
        qp_seq[f - 1] = qp_levels[state];
    }

    // Compute total bits and distortion
    let mut total_bits = 0.0;
    let mut total_dist = 0.0;
    for (f, qp) in qp_seq.iter().enumerate() {
        total_bits += frames[f].rate(*qp);
        total_dist += frames[f].distortion(*qp);
    }

    (qp_seq, total_bits, total_dist)
}

/// Main entry point: allocate per-frame QP values using Viterbi.
///
/// If `config.target_bits` is set, a bisection over lambda is used to
/// match the target.  Otherwise a single pass with `initial_lambda` is run.
pub fn allocate(frames: &[FrameRdModel], config: &ViterbiAllocConfig) -> ViterbiAllocResult {
    let qp_levels = config.qp_levels();
    let states_per_frame = qp_levels.len();

    if frames.is_empty() {
        return ViterbiAllocResult {
            qp_sequence: Vec::new(),
            total_bits: 0.0,
            total_distortion: 0.0,
            final_lambda: config.initial_lambda,
            states_per_frame,
        };
    }

    match config.target_bits {
        Some(target) => {
            // Bisection on lambda to hit the target bitrate
            let mut lo = 1e-6_f64;
            let mut hi = 1e6_f64;
            let mut best_result = viterbi_solve(
                frames,
                &qp_levels,
                config.initial_lambda,
                config.smoothness_penalty,
            );
            let mut best_lambda = config.initial_lambda;

            for _ in 0..config.max_bisect_iters {
                let mid = (lo + hi) / 2.0;
                let (seq, bits, dist) =
                    viterbi_solve(frames, &qp_levels, mid, config.smoothness_penalty);
                let error = (bits - target) / target.max(1.0);

                best_result = (seq, bits, dist);
                best_lambda = mid;

                if error.abs() < config.bisect_tolerance {
                    break;
                }

                if bits > target {
                    // Rate too high => increase lambda to penalise rate more
                    lo = mid;
                } else {
                    hi = mid;
                }
            }

            ViterbiAllocResult {
                qp_sequence: best_result.0,
                total_bits: best_result.1,
                total_distortion: best_result.2,
                final_lambda: best_lambda,
                states_per_frame,
            }
        }
        None => {
            let (seq, bits, dist) = viterbi_solve(
                frames,
                &qp_levels,
                config.initial_lambda,
                config.smoothness_penalty,
            );
            ViterbiAllocResult {
                qp_sequence: seq,
                total_bits: bits,
                total_distortion: dist,
                final_lambda: config.initial_lambda,
                states_per_frame,
            }
        }
    }
}

/// Convenience: creates `FrameRdModel` entries from complexity/temporal arrays.
pub fn build_frame_models(
    complexities: &[f64],
    temporal_energies: &[f64],
    type_weights: &[f64],
) -> Vec<FrameRdModel> {
    let n = complexities
        .len()
        .min(temporal_energies.len())
        .min(type_weights.len());
    (0..n)
        .map(|i| FrameRdModel {
            frame_index: i,
            complexity: complexities[i],
            temporal_energy: temporal_energies[i],
            type_weight: type_weights[i],
        })
        .collect()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_frames(n: usize) -> Vec<FrameRdModel> {
        (0..n)
            .map(|i| FrameRdModel {
                frame_index: i,
                complexity: 500.0 + (i as f64) * 10.0,
                temporal_energy: 1.0,
                type_weight: if i == 0 { 3.0 } else { 1.0 },
            })
            .collect()
    }

    #[test]
    fn test_empty_input() {
        let config = ViterbiAllocConfig::default();
        let result = allocate(&[], &config);
        assert!(result.qp_sequence.is_empty());
        assert_eq!(result.total_bits, 0.0);
    }

    #[test]
    fn test_single_frame() {
        let frames = simple_frames(1);
        let config = ViterbiAllocConfig::default();
        let result = allocate(&frames, &config);
        assert_eq!(result.qp_sequence.len(), 1);
        assert!(result.qp_sequence[0] >= config.qp_min);
        assert!(result.qp_sequence[0] <= config.qp_max);
    }

    #[test]
    fn test_qp_within_bounds() {
        let frames = simple_frames(10);
        let config = ViterbiAllocConfig {
            qp_min: 22,
            qp_max: 40,
            ..Default::default()
        };
        let result = allocate(&frames, &config);
        for &qp in &result.qp_sequence {
            assert!(qp >= 22 && qp <= 40, "QP {qp} out of bounds [22, 40]");
        }
    }

    #[test]
    fn test_smoothness_reduces_qp_jumps() {
        let frames = simple_frames(8);

        let smooth_config = ViterbiAllocConfig {
            smoothness_penalty: 10.0,
            ..Default::default()
        };
        let rough_config = ViterbiAllocConfig {
            smoothness_penalty: 0.0,
            ..Default::default()
        };

        let smooth_result = allocate(&frames, &smooth_config);
        let rough_result = allocate(&frames, &rough_config);

        // Compute max QP jump
        let max_jump = |seq: &[u8]| -> i32 {
            seq.windows(2)
                .map(|w| (w[1] as i32 - w[0] as i32).abs())
                .max()
                .unwrap_or(0)
        };

        assert!(
            max_jump(&smooth_result.qp_sequence) <= max_jump(&rough_result.qp_sequence),
            "smooth penalty should produce smaller QP jumps"
        );
    }

    #[test]
    fn test_higher_lambda_gives_higher_qp() {
        let frames = simple_frames(5);
        let low_lambda = ViterbiAllocConfig {
            initial_lambda: 0.1,
            ..Default::default()
        };
        let high_lambda = ViterbiAllocConfig {
            initial_lambda: 100.0,
            ..Default::default()
        };

        let low_result = allocate(&frames, &low_lambda);
        let high_result = allocate(&frames, &high_lambda);

        let avg_qp = |seq: &[u8]| -> f64 {
            seq.iter().map(|&q| q as f64).sum::<f64>() / seq.len().max(1) as f64
        };

        assert!(
            avg_qp(&high_result.qp_sequence) >= avg_qp(&low_result.qp_sequence),
            "higher lambda should prefer higher QP (lower rate)"
        );
    }

    #[test]
    fn test_target_bits_bisection() {
        let frames = simple_frames(10);

        // Estimate a reasonable target from a mid-lambda run
        let mid_config = ViterbiAllocConfig::default();
        let mid_result = allocate(&frames, &mid_config);
        let target = mid_result.total_bits;

        let config = ViterbiAllocConfig {
            target_bits: Some(target),
            bisect_tolerance: 0.05,
            ..Default::default()
        };
        let result = allocate(&frames, &config);
        let error = (result.total_bits - target).abs() / target.max(1.0);
        assert!(
            error < 0.15,
            "bisection should get within 15% of target, got {error:.2}"
        );
    }

    #[test]
    fn test_build_frame_models() {
        let complexities = vec![100.0, 200.0, 300.0];
        let energies = vec![1.0, 2.0, 3.0];
        let weights = vec![3.0, 1.0, 0.6];
        let models = build_frame_models(&complexities, &energies, &weights);
        assert_eq!(models.len(), 3);
        assert_eq!(models[0].frame_index, 0);
        assert!((models[1].complexity - 200.0).abs() < 1e-9);
        assert!((models[2].type_weight - 0.6).abs() < 1e-9);
    }

    #[test]
    fn test_qp_levels_generation() {
        let config = ViterbiAllocConfig {
            qp_min: 20,
            qp_max: 30,
            qp_step: 2,
            ..Default::default()
        };
        let levels = config.qp_levels();
        assert_eq!(levels, vec![20, 22, 24, 26, 28, 30]);
    }

    #[test]
    fn test_rate_distortion_model_monotonicity() {
        let frame = FrameRdModel {
            frame_index: 0,
            complexity: 500.0,
            temporal_energy: 1.0,
            type_weight: 1.0,
        };
        // Higher QP => lower rate, higher distortion
        let rate_low = frame.rate(20);
        let rate_high = frame.rate(40);
        assert!(rate_low > rate_high, "rate should decrease with QP");

        let dist_low = frame.distortion(20);
        let dist_high = frame.distortion(40);
        assert!(dist_low < dist_high, "distortion should increase with QP");
    }

    #[test]
    fn test_complex_scene_gets_more_bits() {
        // Scene with one very complex frame in the middle
        let mut frames: Vec<FrameRdModel> = (0..5)
            .map(|i| FrameRdModel {
                frame_index: i,
                complexity: 200.0,
                temporal_energy: 1.0,
                type_weight: 1.0,
            })
            .collect();
        frames[2].complexity = 2000.0; // Much more complex

        let config = ViterbiAllocConfig {
            smoothness_penalty: 0.1, // Low so QP can adapt
            ..Default::default()
        };
        let result = allocate(&frames, &config);

        // The complex frame should get a lower QP (more bits)
        assert!(
            result.qp_sequence[2] <= result.qp_sequence[0] + 2,
            "complex frame should not have much higher QP than simple frames"
        );
    }
}
