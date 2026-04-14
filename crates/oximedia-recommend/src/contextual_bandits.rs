//! Contextual bandit algorithms for exploration/exploitation in live recommendations.
//!
//! Implements **LinUCB** (Linear Upper Confidence Bound) — a contextual bandit
//! algorithm that uses per-arm linear models with feature contexts to balance
//! exploration and exploitation. Each arm maintains its own ridge regression
//! model and exploration is driven by confidence intervals.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Dense matrix helpers (small, inline — used only for per-arm A matrices)
// ─────────────────────────────────────────────────────────────────────────────

/// Small square matrix for per-arm LinUCB computation (A = d×d).
#[derive(Debug, Clone)]
struct SmallMatrix {
    data: Vec<f64>,
    dim: usize,
}

impl SmallMatrix {
    /// Identity matrix of size `dim`.
    fn identity(dim: usize) -> Self {
        let mut data = vec![0.0; dim * dim];
        for i in 0..dim {
            data[i * dim + i] = 1.0;
        }
        Self { data, dim }
    }

    fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.dim + c]
    }

    fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.dim + c] = v;
    }

    /// Add outer product: A += x * x^T.
    fn add_outer(&mut self, x: &[f64]) {
        for i in 0..self.dim {
            for j in 0..self.dim {
                let idx = i * self.dim + j;
                self.data[idx] += x[i] * x[j];
            }
        }
    }

    /// Compute A^{-1} via Gauss-Jordan elimination on a copy.
    /// Falls back to identity if singular.
    fn inverse(&self) -> Self {
        let d = self.dim;
        // Augmented matrix [A | I]
        let mut aug = vec![0.0; d * 2 * d];
        for i in 0..d {
            for j in 0..d {
                aug[i * 2 * d + j] = self.get(i, j);
            }
            aug[i * 2 * d + d + i] = 1.0;
        }

        for col in 0..d {
            // Partial pivot
            let mut max_row = col;
            let mut max_val = aug[col * 2 * d + col].abs();
            for row in (col + 1)..d {
                let val = aug[row * 2 * d + col].abs();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
            }
            if max_val < 1e-12 {
                return Self::identity(d);
            }
            if max_row != col {
                for k in 0..(2 * d) {
                    aug.swap(col * 2 * d + k, max_row * 2 * d + k);
                }
            }
            let pivot = aug[col * 2 * d + col];
            for k in 0..(2 * d) {
                aug[col * 2 * d + k] /= pivot;
            }
            for row in 0..d {
                if row == col {
                    continue;
                }
                let factor = aug[row * 2 * d + col];
                for k in 0..(2 * d) {
                    aug[row * 2 * d + k] -= factor * aug[col * 2 * d + k];
                }
            }
        }

        let mut inv = Self::identity(d);
        for i in 0..d {
            for j in 0..d {
                inv.set(i, j, aug[i * 2 * d + d + j]);
            }
        }
        inv
    }

    /// Matrix-vector product.
    fn mul_vec(&self, v: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.dim];
        for i in 0..self.dim {
            let mut s = 0.0;
            for j in 0..self.dim {
                s += self.get(i, j) * v[j];
            }
            out[i] = s;
        }
        out
    }
}

/// Dot product for f64 slices.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// LinUCB arm
// ─────────────────────────────────────────────────────────────────────────────

/// Per-arm model for LinUCB.
#[derive(Debug, Clone)]
struct LinUcbArm {
    /// A matrix (d × d): A = I + sum(x_t x_t^T).
    a_matrix: SmallMatrix,
    /// b vector (d): b = sum(r_t * x_t).
    b_vector: Vec<f64>,
    /// Feature dimension.
    dim: usize,
    /// Number of times this arm has been pulled.
    pulls: u64,
}

impl LinUcbArm {
    fn new(dim: usize) -> Self {
        Self {
            a_matrix: SmallMatrix::identity(dim),
            b_vector: vec![0.0; dim],
            dim,
            pulls: 0,
        }
    }

    /// Compute UCB score: theta^T x + alpha * sqrt(x^T A^{-1} x).
    fn ucb_score(&self, context: &[f64], alpha: f64) -> f64 {
        let a_inv = self.a_matrix.inverse();
        let theta = a_inv.mul_vec(&self.b_vector);
        let expected = dot(&theta, context);

        let a_inv_x = a_inv.mul_vec(context);
        let exploration = dot(context, &a_inv_x).max(0.0).sqrt();

        expected + alpha * exploration
    }

    /// Update model with observed reward.
    fn update(&mut self, context: &[f64], reward: f64) {
        self.a_matrix.add_outer(context);
        for i in 0..self.dim {
            self.b_vector[i] += reward * context[i];
        }
        self.pulls += 1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LinUCB bandit
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for LinUCB contextual bandit.
#[derive(Debug, Clone)]
pub struct LinUcbConfig {
    /// Exploration parameter (higher = more exploration).
    pub alpha: f64,
    /// Feature dimension for context vectors.
    pub feature_dim: usize,
}

impl Default for LinUcbConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            feature_dim: 8,
        }
    }
}

/// LinUCB contextual bandit.
///
/// Each arm has its own linear model. At each round, a context vector
/// is provided and LinUCB selects the arm with the highest upper confidence
/// bound on expected reward.
#[derive(Debug, Clone)]
pub struct LinUcb {
    arms: Vec<LinUcbArm>,
    arm_ids: Vec<String>,
    arm_index: HashMap<String, usize>,
    config: LinUcbConfig,
    total_rounds: u64,
}

impl LinUcb {
    /// Create a new LinUCB bandit with named arms.
    #[must_use]
    pub fn new(arm_ids: Vec<String>, config: LinUcbConfig) -> Self {
        let arms: Vec<LinUcbArm> = (0..arm_ids.len())
            .map(|_| LinUcbArm::new(config.feature_dim))
            .collect();
        let arm_index: HashMap<String, usize> = arm_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), i))
            .collect();
        Self {
            arms,
            arm_ids,
            arm_index,
            config,
            total_rounds: 0,
        }
    }

    /// Select the arm with the highest UCB score given the context.
    ///
    /// Returns the arm ID and index.
    #[must_use]
    pub fn select_arm(&self, context: &[f64]) -> Option<(String, usize)> {
        if self.arms.is_empty() {
            return None;
        }

        let mut best_idx = 0;
        let mut best_score = f64::NEG_INFINITY;

        for (i, arm) in self.arms.iter().enumerate() {
            let score = arm.ucb_score(context, self.config.alpha);
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        self.arm_ids.get(best_idx).map(|id| (id.clone(), best_idx))
    }

    /// Update the model for the given arm with observed reward.
    pub fn update(&mut self, arm_id: &str, context: &[f64], reward: f64) {
        if let Some(&idx) = self.arm_index.get(arm_id) {
            if let Some(arm) = self.arms.get_mut(idx) {
                arm.update(context, reward);
            }
        }
        self.total_rounds += 1;
    }

    /// Update by arm index.
    pub fn update_by_index(&mut self, arm_idx: usize, context: &[f64], reward: f64) {
        if let Some(arm) = self.arms.get_mut(arm_idx) {
            arm.update(context, reward);
        }
        self.total_rounds += 1;
    }

    /// Get the total number of rounds played.
    #[must_use]
    pub fn total_rounds(&self) -> u64 {
        self.total_rounds
    }

    /// Get the number of pulls for a specific arm.
    #[must_use]
    pub fn arm_pulls(&self, arm_id: &str) -> u64 {
        self.arm_index
            .get(arm_id)
            .and_then(|&idx| self.arms.get(idx))
            .map_or(0, |arm| arm.pulls)
    }

    /// Get the number of arms.
    #[must_use]
    pub fn arm_count(&self) -> usize {
        self.arms.len()
    }

    /// Get all arm IDs.
    #[must_use]
    pub fn arm_ids(&self) -> &[String] {
        &self.arm_ids
    }

    /// Get the expected reward for an arm given a context (no exploration bonus).
    #[must_use]
    pub fn expected_reward(&self, arm_id: &str, context: &[f64]) -> Option<f64> {
        let idx = self.arm_index.get(arm_id)?;
        let arm = self.arms.get(*idx)?;
        let a_inv = arm.a_matrix.inverse();
        let theta = a_inv.mul_vec(&arm.b_vector);
        Some(dot(&theta, context))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ContextualBandit trait + wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// Trait for contextual bandit algorithms.
pub trait ContextualBandit {
    /// Select an arm given a context vector. Returns (arm_id, arm_index).
    fn select_arm(&self, context: &[f64]) -> Option<(String, usize)>;

    /// Update the model after observing a reward.
    fn update(&mut self, arm_id: &str, context: &[f64], reward: f64);
}

impl ContextualBandit for LinUcb {
    fn select_arm(&self, context: &[f64]) -> Option<(String, usize)> {
        LinUcb::select_arm(self, context)
    }

    fn update(&mut self, arm_id: &str, context: &[f64], reward: f64) {
        LinUcb::update(self, arm_id, context, reward);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_arms() -> Vec<String> {
        vec!["action".into(), "comedy".into(), "drama".into()]
    }

    fn make_context(vals: &[f64]) -> Vec<f64> {
        vals.to_vec()
    }

    #[test]
    fn test_linucb_creation() {
        let config = LinUcbConfig {
            alpha: 1.0,
            feature_dim: 4,
        };
        let bandit = LinUcb::new(make_arms(), config);
        assert_eq!(bandit.arm_count(), 3);
        assert_eq!(bandit.total_rounds(), 0);
    }

    #[test]
    fn test_linucb_select_arm() {
        let config = LinUcbConfig {
            alpha: 1.0,
            feature_dim: 4,
        };
        let bandit = LinUcb::new(make_arms(), config);
        let ctx = make_context(&[1.0, 0.5, 0.2, 0.8]);
        let result = bandit.select_arm(&ctx);
        assert!(result.is_some());
        let (id, idx) = result.expect("should have result");
        assert!(idx < 3);
        assert!(make_arms().contains(&id));
    }

    #[test]
    fn test_linucb_update_and_learn() {
        let config = LinUcbConfig {
            alpha: 0.5,
            feature_dim: 3,
        };
        let mut bandit = LinUcb::new(make_arms(), config);
        let ctx = make_context(&[1.0, 0.0, 0.0]);

        // Consistently reward "action" with context [1, 0, 0]
        for _ in 0..20 {
            bandit.update("action", &ctx, 1.0);
            bandit.update("comedy", &ctx, 0.1);
            bandit.update("drama", &ctx, 0.1);
        }

        // Now action should be selected for this context
        let (selected, _) = bandit.select_arm(&ctx).expect("should select");
        assert_eq!(selected, "action");
    }

    #[test]
    fn test_linucb_arm_pulls() {
        let config = LinUcbConfig::default();
        let mut bandit = LinUcb::new(make_arms(), config);
        let ctx = make_context(&[1.0; 8]);

        bandit.update("comedy", &ctx, 0.5);
        bandit.update("comedy", &ctx, 0.7);
        bandit.update("drama", &ctx, 0.3);

        assert_eq!(bandit.arm_pulls("comedy"), 2);
        assert_eq!(bandit.arm_pulls("drama"), 1);
        assert_eq!(bandit.arm_pulls("action"), 0);
        assert_eq!(bandit.total_rounds(), 3);
    }

    #[test]
    fn test_linucb_empty_arms() {
        let config = LinUcbConfig::default();
        let bandit = LinUcb::new(vec![], config);
        let ctx = make_context(&[1.0; 8]);
        assert!(bandit.select_arm(&ctx).is_none());
    }

    #[test]
    fn test_linucb_expected_reward() {
        let config = LinUcbConfig {
            alpha: 1.0,
            feature_dim: 2,
        };
        let mut bandit = LinUcb::new(make_arms(), config);
        let ctx = make_context(&[1.0, 0.0]);

        // Before any updates, expected reward should be 0
        let er = bandit.expected_reward("action", &ctx);
        assert!(er.is_some());
        assert!((er.expect("should have reward")).abs() < 1e-10);

        // After updates
        bandit.update("action", &ctx, 5.0);
        let er2 = bandit.expected_reward("action", &ctx);
        assert!(er2.is_some());
        assert!(er2.expect("should have reward") > 0.0);
    }

    #[test]
    fn test_linucb_config_default() {
        let config = LinUcbConfig::default();
        assert_eq!(config.feature_dim, 8);
        assert!((config.alpha - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_linucb_update_by_index() {
        let config = LinUcbConfig {
            alpha: 1.0,
            feature_dim: 3,
        };
        let mut bandit = LinUcb::new(make_arms(), config);
        let ctx = make_context(&[0.5, 0.5, 0.5]);
        bandit.update_by_index(0, &ctx, 1.0);
        assert_eq!(bandit.arm_pulls("action"), 1);
        assert_eq!(bandit.total_rounds(), 1);
    }

    #[test]
    fn test_linucb_arm_ids() {
        let config = LinUcbConfig::default();
        let bandit = LinUcb::new(make_arms(), config);
        assert_eq!(bandit.arm_ids(), &["action", "comedy", "drama"]);
    }

    #[test]
    fn test_contextual_bandit_trait() {
        let config = LinUcbConfig {
            alpha: 1.0,
            feature_dim: 3,
        };
        let mut bandit: Box<dyn ContextualBandit> = Box::new(LinUcb::new(make_arms(), config));
        let ctx = make_context(&[1.0, 0.5, 0.2]);
        let result = bandit.select_arm(&ctx);
        assert!(result.is_some());
        let (id, _) = result.expect("should succeed");
        bandit.update(&id, &ctx, 0.8);
    }

    #[test]
    fn test_linucb_unknown_arm_update() {
        let config = LinUcbConfig::default();
        let mut bandit = LinUcb::new(make_arms(), config);
        let ctx = make_context(&[1.0; 8]);
        // Updating nonexistent arm should not panic
        bandit.update("nonexistent", &ctx, 1.0);
        assert_eq!(bandit.total_rounds(), 1);
    }

    #[test]
    fn test_linucb_exploration_vs_exploitation() {
        let config_explore = LinUcbConfig {
            alpha: 10.0, // High exploration
            feature_dim: 2,
        };
        let config_exploit = LinUcbConfig {
            alpha: 0.01, // Low exploration
            feature_dim: 2,
        };
        let mut bandit_e = LinUcb::new(make_arms(), config_explore);
        let mut bandit_x = LinUcb::new(make_arms(), config_exploit);

        let ctx = make_context(&[1.0, 0.5]);

        // Train both identically
        for _ in 0..10 {
            bandit_e.update("action", &ctx, 1.0);
            bandit_e.update("comedy", &ctx, 0.5);
            bandit_x.update("action", &ctx, 1.0);
            bandit_x.update("comedy", &ctx, 0.5);
        }

        // The exploitative one should definitely pick action
        let (selected_x, _) = bandit_x.select_arm(&ctx).expect("should select");
        assert_eq!(selected_x, "action");
    }

    #[test]
    fn test_small_matrix_inverse_identity() {
        let id = SmallMatrix::identity(3);
        let inv = id.inverse();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (inv.get(i, j) - expected).abs() < 1e-10,
                    "inv[{i},{j}] = {} expected {expected}",
                    inv.get(i, j)
                );
            }
        }
    }
}
