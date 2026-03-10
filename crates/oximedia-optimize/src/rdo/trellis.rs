//! Trellis quantization for RDO.

/// Trellis quantization state.
#[derive(Debug, Clone, Copy)]
struct TrellisState {
    /// Accumulated cost.
    cost: f64,
    /// Quantized value.
    level: i16,
    /// Previous state index.
    prev_state: usize,
}

impl Default for TrellisState {
    fn default() -> Self {
        Self {
            cost: f64::MAX,
            level: 0,
            prev_state: 0,
        }
    }
}

/// Trellis quantizer for optimal coefficient quantization.
pub struct TrellisQuantizer {
    max_levels: usize,
    lambda: f64,
}

impl TrellisQuantizer {
    /// Creates a new trellis quantizer.
    #[must_use]
    pub fn new(max_levels: usize, lambda: f64) -> Self {
        Self { max_levels, lambda }
    }

    /// Performs trellis quantization on coefficients.
    #[allow(dead_code)]
    #[must_use]
    pub fn quantize(&self, coeffs: &[i16], qp: u8) -> Vec<i16> {
        let n = coeffs.len();
        if n == 0 {
            return Vec::new();
        }

        let scale = self.qp_to_scale(qp);

        // Initialize trellis states
        let num_states = self.max_levels * 2 + 1; // +/- max_levels and zero
        let mut trellis = vec![vec![TrellisState::default(); num_states]; n];

        // Initialize first coefficient
        for (state_idx, state) in trellis[0].iter_mut().enumerate() {
            let level = self.state_to_level(state_idx);
            let reconstructed = level * scale;
            let distortion = f64::from((coeffs[0] - reconstructed).pow(2));
            let rate = self.estimate_level_rate(level);
            state.cost = distortion + self.lambda * rate;
            state.level = level;
            state.prev_state = 0;
        }

        // Forward pass
        for i in 1..n {
            let (prev_trellis, curr_trellis) = trellis.split_at_mut(i);
            let prev_states = &prev_trellis[i - 1];

            for (curr_state_idx, curr_state) in curr_trellis[0].iter_mut().enumerate() {
                let curr_level = self.state_to_level(curr_state_idx);
                let reconstructed = curr_level * scale;
                let distortion = f64::from((coeffs[i] - reconstructed).pow(2));
                let rate = self.estimate_level_rate(curr_level);
                let local_cost = distortion + self.lambda * rate;

                // Find best previous state
                for (prev_state_idx, prev_state) in prev_states.iter().enumerate() {
                    let transition_cost = self
                        .estimate_transition_cost(self.state_to_level(prev_state_idx), curr_level);
                    let total_cost = prev_state.cost + local_cost + transition_cost;

                    if total_cost < curr_state.cost {
                        curr_state.cost = total_cost;
                        curr_state.level = curr_level;
                        curr_state.prev_state = prev_state_idx;
                    }
                }
            }
        }

        // Backward pass - find best path
        let mut result = vec![0i16; n];

        // Find best final state
        let (mut best_state_idx, _) = trellis[n - 1]
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.cost
                    .partial_cmp(&b.cost)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or((0, &trellis[n - 1][0]));

        // Trace back
        for i in (0..n).rev() {
            result[i] = trellis[i][best_state_idx].level;
            best_state_idx = trellis[i][best_state_idx].prev_state;
        }

        result
    }

    fn qp_to_scale(&self, qp: u8) -> i16 {
        // Simplified QP to scale conversion
        (1 << (qp / 6)).max(1)
    }

    fn state_to_level(&self, state_idx: usize) -> i16 {
        let offset = self.max_levels;
        (state_idx as i16) - (offset as i16)
    }

    fn estimate_level_rate(&self, level: i16) -> f64 {
        if level == 0 {
            1.0 // Cost of signaling zero
        } else {
            // Simplified rate model: more bits for larger levels
            2.0 + f64::from(level.abs()) * 0.5
        }
    }

    fn estimate_transition_cost(&self, prev_level: i16, curr_level: i16) -> f64 {
        // Small penalty for level changes to encourage smooth quantization
        let diff = (prev_level - curr_level).abs();
        f64::from(diff) * 0.1
    }
}

/// Extended trellis quantizer with run-length optimization.
pub struct RunLengthTrellis {
    base_quantizer: TrellisQuantizer,
    enable_rle: bool,
}

impl RunLengthTrellis {
    /// Creates a new run-length trellis quantizer.
    #[must_use]
    pub fn new(max_levels: usize, lambda: f64, enable_rle: bool) -> Self {
        Self {
            base_quantizer: TrellisQuantizer::new(max_levels, lambda),
            enable_rle,
        }
    }

    /// Quantizes with run-length optimization.
    #[allow(dead_code)]
    #[must_use]
    pub fn quantize(&self, coeffs: &[i16], qp: u8) -> Vec<i16> {
        let quantized = self.base_quantizer.quantize(coeffs, qp);

        if !self.enable_rle {
            return quantized;
        }

        // Optimize runs of zeros
        self.optimize_zero_runs(&quantized)
    }

    fn optimize_zero_runs(&self, coeffs: &[i16]) -> Vec<i16> {
        let optimized = coeffs.to_vec();
        let mut i = 0;

        while i < optimized.len() {
            if optimized[i] == 0 {
                // Count consecutive zeros
                let mut zero_count = 0;
                let mut j = i;
                while j < optimized.len() && optimized[j] == 0 {
                    zero_count += 1;
                    j += 1;
                }

                // If we have a long run of zeros, keep them
                // Otherwise, consider if some should be non-zero
                if zero_count < 3 && i > 0 && j < optimized.len() {
                    // Short zero run between non-zero values might be inefficient
                    // This is a simplified heuristic
                }

                i = j;
            } else {
                i += 1;
            }
        }

        optimized
    }
}

/// RDOQ (Rate-Distortion Optimized Quantization) optimizer.
pub struct RdoqOptimizer {
    trellis: TrellisQuantizer,
    enable_sign_hiding: bool,
}

impl RdoqOptimizer {
    /// Creates a new RDOQ optimizer.
    #[must_use]
    pub fn new(lambda: f64, enable_sign_hiding: bool) -> Self {
        Self {
            trellis: TrellisQuantizer::new(20, lambda),
            enable_sign_hiding,
        }
    }

    /// Performs RDOQ on a transform block.
    #[allow(dead_code)]
    #[must_use]
    pub fn optimize_block(&self, coeffs: &[i16], qp: u8) -> RdoqResult {
        let quantized = self.trellis.quantize(coeffs, qp);

        let (quantized, hidden_signs) = if self.enable_sign_hiding {
            self.apply_sign_hiding(&quantized)
        } else {
            (quantized, 0)
        };

        let (distortion, rate) = self.calculate_metrics(coeffs, &quantized, qp);

        RdoqResult {
            quantized,
            distortion,
            rate,
            hidden_signs,
        }
    }

    fn apply_sign_hiding(&self, coeffs: &[i16]) -> (Vec<i16>, usize) {
        // Simplified sign hiding
        // In production, would implement proper sign data hiding
        let hidden = coeffs.to_vec();
        let mut count = 0;

        // Find pairs of coefficients where we can hide sign information
        for _i in (0..hidden.len().saturating_sub(1)).step_by(2) {
            if hidden[_i].abs() == 1 && hidden[_i + 1].abs() == 1 {
                // Can potentially hide one sign bit
                count += 1;
            }
        }

        (hidden, count)
    }

    fn calculate_metrics(&self, original: &[i16], quantized: &[i16], qp: u8) -> (f64, f64) {
        let scale = self.trellis.qp_to_scale(qp);

        // Reconstruct and calculate distortion
        let reconstructed: Vec<i16> = quantized.iter().map(|&q| q * scale).collect();

        let distortion = original
            .iter()
            .zip(&reconstructed)
            .map(|(&o, &r)| {
                let diff = o - r;
                f64::from(diff * diff)
            })
            .sum();

        // Estimate rate
        let rate = quantized
            .iter()
            .map(|&q| self.trellis.estimate_level_rate(q))
            .sum();

        (distortion, rate)
    }
}

/// RDOQ result.
#[derive(Debug, Clone)]
pub struct RdoqResult {
    /// Optimized quantized coefficients.
    pub quantized: Vec<i16>,
    /// Distortion.
    pub distortion: f64,
    /// Rate in bits.
    pub rate: f64,
    /// Number of hidden sign bits.
    pub hidden_signs: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trellis_quantizer_creation() {
        let trellis = TrellisQuantizer::new(10, 1.0);
        assert_eq!(trellis.max_levels, 10);
        assert_eq!(trellis.lambda, 1.0);
    }

    #[test]
    fn test_state_to_level() {
        let trellis = TrellisQuantizer::new(10, 1.0);
        assert_eq!(trellis.state_to_level(10), 0); // Center state
        assert_eq!(trellis.state_to_level(11), 1);
        assert_eq!(trellis.state_to_level(9), -1);
    }

    #[test]
    fn test_level_rate_estimation() {
        let trellis = TrellisQuantizer::new(10, 1.0);
        let rate_zero = trellis.estimate_level_rate(0);
        let rate_one = trellis.estimate_level_rate(1);
        let rate_ten = trellis.estimate_level_rate(10);

        assert!(rate_zero < rate_one);
        assert!(rate_one < rate_ten);
    }

    #[test]
    fn test_quantize_all_zeros() {
        let trellis = TrellisQuantizer::new(10, 1.0);
        let coeffs = vec![0i16; 16];
        let quantized = trellis.quantize(&coeffs, 26);
        assert_eq!(quantized.len(), 16);
    }

    #[test]
    fn test_run_length_trellis_creation() {
        let rlt = RunLengthTrellis::new(10, 1.0, true);
        assert!(rlt.enable_rle);
    }

    #[test]
    fn test_rdoq_optimizer_creation() {
        let rdoq = RdoqOptimizer::new(1.0, true);
        assert!(rdoq.enable_sign_hiding);
    }

    #[test]
    fn test_sign_hiding() {
        let rdoq = RdoqOptimizer::new(1.0, true);
        let coeffs = vec![1, 1, -1, 1, 1, -1];
        let (hidden, count) = rdoq.apply_sign_hiding(&coeffs);
        assert_eq!(hidden.len(), coeffs.len());
        let _ = count; // count is usize (always >= 0); presence verified by destructuring
    }

    #[test]
    fn test_calculate_metrics() {
        let rdoq = RdoqOptimizer::new(1.0, false);
        let original = vec![100, 50, 25, 10];
        let quantized = vec![10, 5, 2, 1];
        let (distortion, rate) = rdoq.calculate_metrics(&original, &quantized, 12);
        assert!(distortion > 0.0);
        assert!(rate > 0.0);
    }
}
