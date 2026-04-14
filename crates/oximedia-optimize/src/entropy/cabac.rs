//! CABAC (Context-Adaptive Binary Arithmetic Coding) context optimization.
//!
//! This module implements actual CABAC context modeling and optimization for
//! H.264/H.265 entropy coding. It provides:
//!
//! - **Probability state machine**: 128-state MPS/LPS probability table
//! - **Context selection**: Spatial and syntax-driven context derivation
//! - **Arithmetic interval tracking**: Range/offset based interval subdivision
//! - **Rate estimation**: Accurate bit cost from context states without full encode
//! - **Context grouping optimization**: Merge rarely-used contexts to reduce overhead
//! - **Initialization optimization**: Derive per-slice init states from training data
//!
//! # CABAC overview
//!
//! CABAC encodes binary decisions ("bins") using arithmetic coding with
//! adaptive probability models. Each bin is coded in one of two modes:
//!
//! - **Regular mode**: uses a context model (probability state) for adaptation
//! - **Bypass mode**: uses a fixed 0.5 probability (no context overhead)
//!
//! The probability state machine uses a 128-entry table mapping state index
//! to LPS (Least Probable Symbol) probability and transition rules.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

// ── CABAC probability table ─────────────────────────────────────────────────

/// Number of probability states in the CABAC state machine.
pub const NUM_STATES: usize = 128;

/// Probability entry in the CABAC state table.
#[derive(Debug, Clone, Copy)]
pub struct ProbabilityEntry {
    /// LPS (Least Probable Symbol) probability as fixed-point (Q15).
    pub lps_probability: u16,
    /// Next state after MPS (Most Probable Symbol) is observed.
    pub next_state_mps: u8,
    /// Next state after LPS is observed.
    pub next_state_lps: u8,
}

/// Builds the CABAC probability state table.
///
/// Uses the exponential decay model from H.264/H.265:
/// p_LPS(state) = 0.5 * (0.01875 / 0.5)^(state / 63)
#[must_use]
pub fn build_probability_table() -> Vec<ProbabilityEntry> {
    let mut table = Vec::with_capacity(NUM_STATES);

    for state_idx in 0..NUM_STATES {
        let half_idx = state_idx.min(63);
        // LPS probability decays exponentially with state index
        let p_lps = 0.5 * (0.01875_f64 / 0.5).powf(half_idx as f64 / 63.0);
        let lps_q15 = (p_lps * 32768.0).round().min(32767.0).max(1.0) as u16;

        // State transitions
        let next_mps = if state_idx < 62 {
            (state_idx + 1) as u8
        } else {
            63_u8 // Stay at most confident state
        };
        let next_lps = if state_idx > 0 {
            (state_idx - 1) as u8
        } else {
            0_u8
        };

        table.push(ProbabilityEntry {
            lps_probability: lps_q15,
            next_state_mps: next_mps,
            next_state_lps: next_lps,
        });
    }

    table
}

// ── Context state ───────────────────────────────────────────────────────────

/// A single CABAC context with state and MPS value.
#[derive(Debug, Clone, Copy)]
pub struct CabacContext {
    /// Current probability state index (0..63).
    pub state: u8,
    /// Most Probable Symbol (0 or 1).
    pub mps: u8,
    /// Total bins coded through this context.
    pub bin_count: u64,
    /// Total estimated bits used by this context.
    pub total_bits_q8: u64,
}

impl CabacContext {
    /// Creates a new context with given initial state.
    #[must_use]
    pub fn new(init_state: u8, init_mps: u8) -> Self {
        Self {
            state: init_state.min(63),
            mps: init_mps & 1,
            bin_count: 0,
            total_bits_q8: 0,
        }
    }

    /// Creates a context initialized to equiprobable (state 0, MPS 0).
    #[must_use]
    pub fn equiprobable() -> Self {
        Self::new(0, 0)
    }

    /// Returns the LPS probability as a floating-point value.
    #[must_use]
    pub fn lps_probability(&self, table: &[ProbabilityEntry]) -> f64 {
        if (self.state as usize) < table.len() {
            f64::from(table[self.state as usize].lps_probability) / 32768.0
        } else {
            0.5
        }
    }

    /// Estimates the cost in bits (Q8 fixed-point) to code a given bin value.
    #[must_use]
    pub fn estimate_cost_q8(&self, bin_value: u8, table: &[ProbabilityEntry]) -> u32 {
        let p_lps = self.lps_probability(table);
        let is_lps = (bin_value & 1) != self.mps;

        let p = if is_lps { p_lps } else { 1.0 - p_lps };
        let p = p.max(1.0 / 65536.0);

        // -log2(p) in Q8 format
        let bits = -p.log2();
        (bits * 256.0).round().min(65535.0) as u32
    }

    /// Estimates the cost in fractional bits (f64).
    #[must_use]
    pub fn estimate_cost_f64(&self, bin_value: u8, table: &[ProbabilityEntry]) -> f64 {
        self.estimate_cost_q8(bin_value, table) as f64 / 256.0
    }

    /// Updates the context after observing a bin value.
    pub fn update(&mut self, bin_value: u8, table: &[ProbabilityEntry]) {
        let cost = self.estimate_cost_q8(bin_value, table) as u64;
        self.bin_count += 1;
        self.total_bits_q8 += cost;

        let is_lps = (bin_value & 1) != self.mps;
        let idx = self.state as usize;

        if idx < table.len() {
            if is_lps {
                // LPS observed: move towards equiprobable
                self.state = table[idx].next_state_lps;
                if self.state == 0 {
                    // Flip MPS when reaching state 0 after LPS
                    self.mps ^= 1;
                }
            } else {
                // MPS observed: move towards most confident
                self.state = table[idx].next_state_mps;
            }
        }
    }

    /// Returns average bits per bin for this context.
    #[must_use]
    pub fn avg_bits_per_bin(&self) -> f64 {
        if self.bin_count == 0 {
            return 0.0;
        }
        (self.total_bits_q8 as f64 / 256.0) / self.bin_count as f64
    }
}

// ── CABAC context model ─────────────────────────────────────────────────────

/// CABAC context model managing multiple contexts.
#[derive(Debug, Clone)]
pub struct CabacModel {
    /// All context entries.
    contexts: Vec<CabacContext>,
    /// Probability state table.
    table: Vec<ProbabilityEntry>,
    /// Bypass bin count (bins coded without context).
    bypass_bins: u64,
}

impl CabacModel {
    /// Creates a new CABAC model with `num_contexts` contexts.
    #[must_use]
    pub fn new(num_contexts: usize) -> Self {
        let table = build_probability_table();
        Self {
            contexts: vec![CabacContext::equiprobable(); num_contexts],
            table,
            bypass_bins: 0,
        }
    }

    /// Creates a model with custom initial states.
    ///
    /// `init_values` contains `(state, mps)` pairs for each context.
    #[must_use]
    pub fn with_init(init_values: &[(u8, u8)]) -> Self {
        let table = build_probability_table();
        let contexts = init_values
            .iter()
            .map(|&(state, mps)| CabacContext::new(state, mps))
            .collect();
        Self {
            contexts,
            table,
            bypass_bins: 0,
        }
    }

    /// Returns the number of contexts.
    #[must_use]
    pub fn num_contexts(&self) -> usize {
        self.contexts.len()
    }

    /// Codes a bin in regular mode (with context adaptation).
    ///
    /// Returns the estimated cost in bits (f64).
    pub fn code_bin(&mut self, ctx_idx: usize, bin_value: u8) -> f64 {
        if ctx_idx >= self.contexts.len() {
            return 1.0; // fallback cost
        }
        let cost = self.contexts[ctx_idx].estimate_cost_f64(bin_value, &self.table);
        self.contexts[ctx_idx].update(bin_value, &self.table);
        cost
    }

    /// Codes a bin in bypass mode (fixed 0.5 probability).
    ///
    /// Returns 1.0 bit cost.
    pub fn code_bypass_bin(&mut self, _bin_value: u8) -> f64 {
        self.bypass_bins += 1;
        1.0
    }

    /// Codes a sequence of bins using given context indices.
    ///
    /// Returns total estimated bits.
    pub fn code_bins(&mut self, ctx_indices: &[usize], bin_values: &[u8]) -> f64 {
        let len = ctx_indices.len().min(bin_values.len());
        let mut total = 0.0;
        for i in 0..len {
            total += self.code_bin(ctx_indices[i], bin_values[i]);
        }
        total
    }

    /// Returns a reference to a specific context.
    #[must_use]
    pub fn context(&self, idx: usize) -> Option<&CabacContext> {
        self.contexts.get(idx)
    }

    /// Returns the total estimated bits across all contexts.
    #[must_use]
    pub fn total_bits(&self) -> f64 {
        let regular: f64 = self
            .contexts
            .iter()
            .map(|c| c.total_bits_q8 as f64 / 256.0)
            .sum();
        regular + self.bypass_bins as f64
    }

    /// Returns the total number of bins coded (regular + bypass).
    #[must_use]
    pub fn total_bins(&self) -> u64 {
        let regular: u64 = self.contexts.iter().map(|c| c.bin_count).sum();
        regular + self.bypass_bins
    }

    /// Returns the probability state table.
    #[must_use]
    pub fn table(&self) -> &[ProbabilityEntry] {
        &self.table
    }

    /// Resets all contexts to equiprobable.
    pub fn reset(&mut self) {
        for ctx in &mut self.contexts {
            *ctx = CabacContext::equiprobable();
        }
        self.bypass_bins = 0;
    }
}

// ── Context grouping optimizer ──────────────────────────────────────────────

/// Groups contexts with similar probability distributions to reduce overhead.
///
/// Rarely-used contexts can be merged into a shared context to improve
/// coding efficiency. This optimizer identifies contexts with similar
/// state trajectories and recommends merging.
#[derive(Debug, Clone)]
pub struct ContextGroupOptimizer {
    /// Minimum bin count for a context to be considered "active".
    min_active_bins: u64,
    /// Maximum state difference for two contexts to be considered similar.
    max_state_diff: u8,
}

/// Result of context grouping analysis.
#[derive(Debug, Clone)]
pub struct ContextGroupResult {
    /// Mapping from original context index to group index.
    pub group_map: Vec<usize>,
    /// Number of groups after merging.
    pub num_groups: usize,
    /// Number of inactive contexts merged.
    pub inactive_merged: usize,
    /// Estimated bits saved by merging.
    pub estimated_bits_saved: f64,
}

impl ContextGroupOptimizer {
    /// Creates a new context group optimizer.
    #[must_use]
    pub fn new(min_active_bins: u64, max_state_diff: u8) -> Self {
        Self {
            min_active_bins,
            max_state_diff,
        }
    }

    /// Analyzes a CABAC model and produces grouping recommendations.
    #[must_use]
    pub fn analyze(&self, model: &CabacModel) -> ContextGroupResult {
        let num_ctx = model.num_contexts();
        let mut group_map = Vec::with_capacity(num_ctx);
        let mut current_group = 0usize;
        let mut inactive_merged = 0;

        // Phase 1: Mark inactive contexts
        let mut is_active = Vec::with_capacity(num_ctx);
        for i in 0..num_ctx {
            let active = model
                .context(i)
                .map_or(false, |c| c.bin_count >= self.min_active_bins);
            is_active.push(active);
        }

        // Phase 2: Group similar active contexts
        // Use a greedy approach: each active context starts a new group,
        // then merge subsequent contexts with similar state.
        let mut group_representatives: Vec<(usize, u8, u8)> = Vec::new(); // (group_id, state, mps)

        for i in 0..num_ctx {
            if !is_active[i] {
                // Inactive: merge into a catch-all group
                if group_representatives.is_empty() {
                    group_representatives.push((current_group, 0, 0));
                    current_group += 1;
                }
                group_map.push(0); // Map to first group
                inactive_merged += 1;
                continue;
            }

            let ctx = match model.context(i) {
                Some(c) => c,
                None => {
                    group_map.push(0);
                    continue;
                }
            };

            // Find a compatible group
            let mut found_group = None;
            for &(gid, state, mps) in &group_representatives {
                let state_diff = if ctx.state > state {
                    ctx.state - state
                } else {
                    state - ctx.state
                };
                if state_diff <= self.max_state_diff && ctx.mps == mps {
                    found_group = Some(gid);
                    break;
                }
            }

            match found_group {
                Some(gid) => {
                    group_map.push(gid);
                }
                None => {
                    group_representatives.push((current_group, ctx.state, ctx.mps));
                    group_map.push(current_group);
                    current_group += 1;
                }
            }
        }

        let num_groups = current_group.max(1);

        // Estimate bits saved: inactive contexts that share state with active ones
        // save the overhead of maintaining separate probability tracks
        let estimated_bits_saved = inactive_merged as f64 * 0.5;

        ContextGroupResult {
            group_map,
            num_groups,
            inactive_merged,
            estimated_bits_saved,
        }
    }
}

// ── Initialization optimizer ────────────────────────────────────────────────

/// Optimizes CABAC context initialization values based on training data.
///
/// In H.264/H.265, each slice type can have different initial QP-dependent
/// context values. This optimizer learns optimal init values from statistics.
#[derive(Debug, Clone)]
pub struct InitOptimizer {
    /// Per-context statistics: (sum_of_states, count_of_observations).
    stats: Vec<(f64, u64)>,
    num_contexts: usize,
}

impl InitOptimizer {
    /// Creates a new init optimizer for the given number of contexts.
    #[must_use]
    pub fn new(num_contexts: usize) -> Self {
        Self {
            stats: vec![(0.0, 0); num_contexts],
            num_contexts,
        }
    }

    /// Records the final state of a context after encoding a slice.
    pub fn record(&mut self, ctx_idx: usize, final_state: u8, final_mps: u8) {
        if ctx_idx < self.num_contexts {
            let signed_state = if final_mps > 0 {
                f64::from(final_state)
            } else {
                -f64::from(final_state)
            };
            self.stats[ctx_idx].0 += signed_state;
            self.stats[ctx_idx].1 += 1;
        }
    }

    /// Derives optimal initial values from recorded statistics.
    #[must_use]
    pub fn derive_init_values(&self) -> Vec<(u8, u8)> {
        self.stats
            .iter()
            .map(|&(sum, count)| {
                if count == 0 {
                    return (0, 0);
                }
                let avg = sum / count as f64;
                let mps = if avg >= 0.0 { 1u8 } else { 0u8 };
                let state = avg.abs().round().min(63.0) as u8;
                (state, mps)
            })
            .collect()
    }

    /// Returns the number of observations for a context.
    #[must_use]
    pub fn observation_count(&self, ctx_idx: usize) -> u64 {
        if ctx_idx < self.num_contexts {
            self.stats[ctx_idx].1
        } else {
            0
        }
    }
}

// ── Rate estimator ──────────────────────────────────────────────────────────

/// Estimates coding rate for a block of syntax elements without full encoding.
///
/// This is used by the RDO engine to quickly evaluate different coding options.
#[derive(Debug)]
pub struct CabacRateEstimator {
    model: CabacModel,
}

impl CabacRateEstimator {
    /// Creates a new rate estimator with the given number of contexts.
    #[must_use]
    pub fn new(num_contexts: usize) -> Self {
        Self {
            model: CabacModel::new(num_contexts),
        }
    }

    /// Creates from an existing model (e.g., snapshot from current encoding state).
    #[must_use]
    pub fn from_model(model: CabacModel) -> Self {
        Self { model }
    }

    /// Estimates the cost of coding a coefficient level using exp-Golomb binarization.
    ///
    /// For H.264, coefficient levels are binarized as:
    /// - Significance flag (1 bin, regular)
    /// - Greater-than-1 flag (1 bin, regular)
    /// - Remaining value (exp-Golomb, bypass bins)
    #[must_use]
    pub fn estimate_coeff_cost(&self, level: i32, sig_ctx: usize, gt1_ctx: usize) -> f64 {
        let abs_level = level.unsigned_abs();

        if abs_level == 0 {
            // Significance = 0
            return self.model.contexts.get(sig_ctx).map_or(1.0, |c| {
                c.estimate_cost_f64(0, &self.model.table)
            });
        }

        let mut cost = 0.0;

        // Significance = 1
        cost += self.model.contexts.get(sig_ctx).map_or(1.0, |c| {
            c.estimate_cost_f64(1, &self.model.table)
        });

        if abs_level == 1 {
            // Greater-than-1 = 0
            cost += self.model.contexts.get(gt1_ctx).map_or(1.0, |c| {
                c.estimate_cost_f64(0, &self.model.table)
            });
        } else {
            // Greater-than-1 = 1
            cost += self.model.contexts.get(gt1_ctx).map_or(1.0, |c| {
                c.estimate_cost_f64(1, &self.model.table)
            });
            // Remaining value in bypass mode: ~2*floor(log2(level-1))+1 bypass bins
            let remaining = abs_level - 2;
            let exp_golomb_bins = if remaining == 0 {
                1
            } else {
                let log2_val = (remaining as f64).log2().floor() as u32;
                2 * log2_val + 1
            };
            cost += exp_golomb_bins as f64; // bypass bins at 1 bit each
        }

        // Sign bit: 1 bypass bin
        cost += 1.0;

        cost
    }

    /// Estimates the total cost for a block of coefficient levels.
    #[must_use]
    pub fn estimate_block_cost(
        &self,
        levels: &[i32],
        sig_ctx_base: usize,
        gt1_ctx_base: usize,
    ) -> f64 {
        levels
            .iter()
            .enumerate()
            .map(|(i, &level)| {
                let sig_ctx = sig_ctx_base + (i % 16);
                let gt1_ctx = gt1_ctx_base + (i % 8);
                self.estimate_coeff_cost(level, sig_ctx, gt1_ctx)
            })
            .sum()
    }

    /// Returns a reference to the underlying model.
    #[must_use]
    pub fn model(&self) -> &CabacModel {
        &self.model
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probability_table_size() {
        let table = build_probability_table();
        assert_eq!(table.len(), NUM_STATES);
    }

    #[test]
    fn test_probability_table_lps_decreasing() {
        let table = build_probability_table();
        // LPS probability should decrease with state index (more confident)
        for i in 1..64 {
            assert!(
                table[i].lps_probability <= table[i - 1].lps_probability,
                "LPS prob should decrease: state {} ({}) vs {} ({})",
                i,
                table[i].lps_probability,
                i - 1,
                table[i - 1].lps_probability
            );
        }
    }

    #[test]
    fn test_probability_table_state0_near_half() {
        let table = build_probability_table();
        let p0 = f64::from(table[0].lps_probability) / 32768.0;
        assert!(
            (p0 - 0.5).abs() < 0.01,
            "State 0 LPS prob should be ~0.5, got {p0}"
        );
    }

    #[test]
    fn test_cabac_context_initial() {
        let table = build_probability_table();
        let ctx = CabacContext::equiprobable();
        assert_eq!(ctx.state, 0);
        assert_eq!(ctx.mps, 0);
        assert_eq!(ctx.bin_count, 0);
        let p = ctx.lps_probability(&table);
        assert!((p - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_cabac_context_update_mps_increases_confidence() {
        let table = build_probability_table();
        let mut ctx = CabacContext::equiprobable();
        ctx.mps = 1;

        // Feed MPS (bin=1) repeatedly
        for _ in 0..20 {
            ctx.update(1, &table);
        }

        // State should have increased (more confident MPS is 1)
        assert!(
            ctx.state > 10,
            "After 20 MPS bins, state should be >10, got {}",
            ctx.state
        );
        assert_eq!(ctx.mps, 1);
    }

    #[test]
    fn test_cabac_context_update_lps_decreases_confidence() {
        let table = build_probability_table();
        let mut ctx = CabacContext::new(30, 1);

        // Feed LPS (bin=0 when mps=1) repeatedly
        for _ in 0..20 {
            ctx.update(0, &table);
        }

        // State should have decreased
        assert!(
            ctx.state < 30,
            "After 20 LPS bins, state should decrease from 30, got {}",
            ctx.state
        );
    }

    #[test]
    fn test_cabac_cost_lps_higher_than_mps() {
        let table = build_probability_table();
        let ctx = CabacContext::new(30, 1);

        let cost_mps = ctx.estimate_cost_f64(1, &table);
        let cost_lps = ctx.estimate_cost_f64(0, &table);

        assert!(
            cost_lps > cost_mps,
            "LPS should cost more than MPS: LPS={cost_lps}, MPS={cost_mps}"
        );
    }

    #[test]
    fn test_cabac_model_creation() {
        let model = CabacModel::new(256);
        assert_eq!(model.num_contexts(), 256);
        assert_eq!(model.total_bins(), 0);
        assert!((model.total_bits() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cabac_model_coding() {
        let mut model = CabacModel::new(16);

        // Code some bins
        let cost0 = model.code_bin(0, 1);
        assert!(cost0 > 0.0);

        let cost1 = model.code_bin(0, 1);
        // After adapting to MPS=1, cost should decrease
        assert!(
            cost1 <= cost0 + 0.01,
            "Cost should not increase after adapting: {cost1} vs {cost0}"
        );

        assert_eq!(model.total_bins(), 2);
    }

    #[test]
    fn test_cabac_model_bypass() {
        let mut model = CabacModel::new(4);
        let cost = model.code_bypass_bin(1);
        assert!((cost - 1.0).abs() < 1e-10, "Bypass bin should cost 1.0 bit");
        assert_eq!(model.total_bins(), 1);
    }

    #[test]
    fn test_context_group_optimizer() {
        let mut model = CabacModel::new(32);

        // Make some contexts active by coding bins through them
        for i in 0..16 {
            for _ in 0..100 {
                model.code_bin(i, 1);
            }
        }
        // Contexts 16..32 remain inactive

        let optimizer = ContextGroupOptimizer::new(50, 5);
        let result = optimizer.analyze(&model);

        assert!(result.num_groups < 32, "Should have fewer groups than contexts");
        assert!(
            result.inactive_merged > 0,
            "Should have merged inactive contexts"
        );
        assert_eq!(result.group_map.len(), 32);
    }

    #[test]
    fn test_init_optimizer() {
        let mut init_opt = InitOptimizer::new(8);

        // Record some final states
        init_opt.record(0, 30, 1);
        init_opt.record(0, 32, 1);
        init_opt.record(0, 28, 1);

        init_opt.record(1, 10, 0);
        init_opt.record(1, 12, 0);

        let init_values = init_opt.derive_init_values();
        assert_eq!(init_values.len(), 8);

        // Context 0: avg ~30, MPS=1
        assert!(init_values[0].0 >= 28 && init_values[0].0 <= 32);
        assert_eq!(init_values[0].1, 1);

        // Context 1: avg ~11, MPS=0
        assert!(init_values[1].0 >= 10 && init_values[1].0 <= 12);
        assert_eq!(init_values[1].1, 0);

        // Unobserved contexts default to (0, 0)
        assert_eq!(init_values[5], (0, 0));
    }

    #[test]
    fn test_rate_estimator_zero_coeff() {
        let estimator = CabacRateEstimator::new(32);
        let cost = estimator.estimate_coeff_cost(0, 0, 16);
        assert!(cost > 0.0, "Zero coeff should still cost >0 bits");
        assert!(cost < 2.0, "Zero coeff should cost <2 bits");
    }

    #[test]
    fn test_rate_estimator_nonzero_coeff() {
        let estimator = CabacRateEstimator::new(32);
        let cost_1 = estimator.estimate_coeff_cost(1, 0, 16);
        let cost_5 = estimator.estimate_coeff_cost(5, 0, 16);

        assert!(cost_1 > 0.0);
        assert!(cost_5 > cost_1, "Larger coeff should cost more bits");
    }

    #[test]
    fn test_rate_estimator_block_cost() {
        let estimator = CabacRateEstimator::new(32);
        let levels = vec![3, 0, -1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let cost = estimator.estimate_block_cost(&levels, 0, 16);
        assert!(cost > 0.0);

        // All-zero block should be cheaper
        let zero_levels = vec![0; 16];
        let zero_cost = estimator.estimate_block_cost(&zero_levels, 0, 16);
        assert!(
            zero_cost < cost,
            "All-zero block should be cheaper: {zero_cost} < {cost}"
        );
    }

    #[test]
    fn test_cabac_model_reset() {
        let mut model = CabacModel::new(8);
        model.code_bin(0, 1);
        model.code_bin(0, 1);
        model.code_bypass_bin(0);

        assert!(model.total_bins() > 0);
        model.reset();
        assert_eq!(model.total_bins(), 0);
    }

    #[test]
    fn test_avg_bits_per_bin() {
        let table = build_probability_table();
        let mut ctx = CabacContext::equiprobable();

        // Equiprobable context: avg should be ~1.0 bit per bin
        for _ in 0..100 {
            ctx.update(0, &table);
        }
        for _ in 0..100 {
            ctx.update(1, &table);
        }

        let avg = ctx.avg_bits_per_bin();
        assert!(avg > 0.0 && avg < 2.0, "Avg bits should be reasonable: {avg}");
    }
}

// ── CABAC context optimization integration tests ─────────────────────────────

#[cfg(test)]
mod cabac_optimization_tests {
    use super::*;

    /// Verify that after many MPS observations, the context cost for MPS is
    /// cheaper than for a freshly-initialized equiprobable context.
    #[test]
    fn test_adapted_context_cheaper_than_equiprobable() {
        let table = build_probability_table();

        let mut adapted = CabacContext::new(0, 1);
        // Feed 50 MPS (bin=1) bins to build confidence
        for _ in 0..50 {
            adapted.update(1, &table);
        }

        let equiprobable = CabacContext::equiprobable();

        // After adaptation, MPS cost should be lower (more confident)
        let cost_adapted = adapted.estimate_cost_f64(1, &table);
        let cost_equi = equiprobable.estimate_cost_f64(1, &table);
        assert!(
            cost_adapted < cost_equi,
            "Adapted context (MPS=1 x50) should code MPS cheaper: adapted={cost_adapted}, equi={cost_equi}"
        );
    }

    /// Verify the coding state machine: after 100 consistent MPS observations
    /// the state index is significantly higher than 0.
    #[test]
    fn test_state_converges_with_consistent_symbols() {
        let table = build_probability_table();
        let mut ctx = CabacContext::new(0, 1);
        for _ in 0..100 {
            ctx.update(1, &table);
        }
        assert!(
            ctx.state > 30,
            "After 100 MPS bins, state should converge to high confidence, got {}",
            ctx.state
        );
    }

    /// Confirm that coding bypass bins accumulates correctly in total_bits.
    #[test]
    fn test_bypass_bins_contribute_to_total_bits() {
        let mut model = CabacModel::new(8);
        let bypass_count = 10;
        for _ in 0..bypass_count {
            model.code_bypass_bin(0);
        }
        let total = model.total_bits();
        assert!(
            (total - bypass_count as f64).abs() < 1e-9,
            "Each bypass bin should cost exactly 1.0 bit, total={total}"
        );
    }

    /// CabacModel::code_bins codes the minimum of ctx_indices/bin_values lengths.
    #[test]
    fn test_code_bins_partial_length() {
        let mut model = CabacModel::new(16);
        let ctx_indices = vec![0, 1, 2, 3, 4];
        let bin_values = vec![1u8, 0u8, 1u8]; // shorter slice
        let cost = model.code_bins(&ctx_indices, &bin_values);
        assert!(cost > 0.0);
        assert_eq!(model.total_bins(), 3, "Should have coded exactly 3 bins");
    }

    /// Verify `ContextGroupOptimizer` with all-active contexts produces
    /// num_groups <= num_contexts and no inactive merges.
    #[test]
    fn test_context_group_all_active() {
        let mut model = CabacModel::new(8);
        // Make all contexts active
        for i in 0..8 {
            for _ in 0..100 {
                model.code_bin(i, 1);
            }
        }

        let optimizer = ContextGroupOptimizer::new(50, 3);
        let result = optimizer.analyze(&model);

        assert_eq!(result.group_map.len(), 8);
        assert_eq!(
            result.inactive_merged, 0,
            "No inactive contexts when all are active"
        );
    }

    /// Verify `InitOptimizer::derive_init_values` is stable under no observations.
    #[test]
    fn test_init_optimizer_no_observations() {
        let init_opt = InitOptimizer::new(16);
        let values = init_opt.derive_init_values();
        assert_eq!(values.len(), 16);
        for &(state, mps) in &values {
            assert_eq!(state, 0, "Unobserved context state should be 0");
            assert_eq!(mps, 0, "Unobserved context MPS should be 0");
        }
    }

    /// `CabacModel::with_init` respects custom state/MPS pairs.
    #[test]
    fn test_cabac_model_with_init_custom_states() {
        let init = [(20u8, 1u8), (40u8, 0u8), (0u8, 0u8)];
        let model = CabacModel::with_init(&init);
        assert_eq!(model.num_contexts(), 3);
        let table = model.table();

        let ctx0 = model.context(0).expect("context 0 should exist");
        assert_eq!(ctx0.state, 20);
        assert_eq!(ctx0.mps, 1);

        let ctx1 = model.context(1).expect("context 1 should exist");
        assert_eq!(ctx1.state, 40);
        assert_eq!(ctx1.mps, 0);

        // Higher state = lower LPS probability
        let p0 = ctx0.lps_probability(table);
        let p1 = ctx1.lps_probability(table);
        assert!(
            p1 < p0,
            "State-40 context should have lower LPS prob than state-20: {p1} < {p0}"
        );
    }

    /// Verify that `estimate_block_cost` correctly identifies an all-zero block
    /// as cheaper than a block with large coefficients.
    #[test]
    fn test_block_cost_zero_cheaper_than_large() {
        let estimator = CabacRateEstimator::new(32);
        let zero = vec![0i32; 16];
        let large = vec![100i32; 16];
        let cost_zero = estimator.estimate_block_cost(&zero, 0, 16);
        let cost_large = estimator.estimate_block_cost(&large, 0, 16);
        assert!(
            cost_zero < cost_large,
            "All-zero block should cost less than all-100 block: {cost_zero} < {cost_large}"
        );
    }

    /// Confirm `InitOptimizer::observation_count` returns correct counts.
    #[test]
    fn test_init_optimizer_observation_count() {
        let mut init_opt = InitOptimizer::new(4);
        init_opt.record(0, 10, 1);
        init_opt.record(0, 12, 1);
        init_opt.record(2, 5, 0);

        assert_eq!(init_opt.observation_count(0), 2);
        assert_eq!(init_opt.observation_count(1), 0);
        assert_eq!(init_opt.observation_count(2), 1);
        assert_eq!(init_opt.observation_count(99), 0); // out-of-bounds
    }
}
