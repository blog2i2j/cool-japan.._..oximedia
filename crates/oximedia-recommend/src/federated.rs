//! Federated learning support for collaborative recommendation models.
//!
//! Federated learning enables training collaborative filtering models across
//! many users without centralizing raw interaction data.  Each participant
//! computes a local gradient update from its private data, adds calibrated
//! Gaussian noise for differential privacy, and uploads only the **noised
//! gradient** to the aggregator.  The aggregator applies FedAvg to merge
//! gradients into a global model update.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────┐
//! │  FederatedAggregator                             │
//! │  ┌──────────────┐   FedAvg    ┌───────────────┐ │
//! │  │ GlobalModel  │ ◄──────── │ LocalUpdate[] │ │
//! │  └──────────────┘            └───────────────┘ │
//! └──────────────────────────────────────────────────┘
//!         ▲
//!  upload noise-masked gradient
//!         │
//! ┌───────┴────────────────────────┐
//! │  FederatedClient (per device)  │
//! │  private interaction history   │
//! └────────────────────────────────┘
//! ```
//!
//! # Differential Privacy
//!
//! Each client adds Gaussian noise with standard deviation
//! `σ = dp_sensitivity * dp_noise_scale` to its gradient before upload,
//! ensuring (ε, δ)-differential privacy.

#![allow(dead_code)]

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// LCG helpers (no external RNG dep)
// ─────────────────────────────────────────────────────────────────────────────

/// Advance a 64-bit LCG state.
#[inline]
fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
}

/// Map LCG state to [0, 1).
#[inline]
fn lcg_f64(state: u64) -> f64 {
    let s = lcg_next(state);
    (s >> 11) as f64 / (1u64 << 53) as f64
}

/// Box-Muller transform: draw one N(0,1) sample given a LCG seed.
fn normal_sample(seed: u64) -> f64 {
    let s1 = lcg_next(seed);
    let s2 = lcg_next(s1);
    let u1 = lcg_f64(s1).max(1e-15);
    let u2 = lcg_f64(s2);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ─────────────────────────────────────────────────────────────────────────────
// Differential privacy config
// ─────────────────────────────────────────────────────────────────────────────

/// Differential-privacy parameters for gradient noising.
#[derive(Debug, Clone)]
pub struct DpConfig {
    /// L2 sensitivity of the gradient (clip norm).
    pub sensitivity: f64,
    /// Noise scale multiplier (σ = sensitivity * noise_scale).
    pub noise_scale: f64,
    /// Gradient L2-norm clipping threshold.
    pub clip_norm: f64,
    /// Whether to enable DP noising.
    pub enabled: bool,
}

impl Default for DpConfig {
    fn default() -> Self {
        Self {
            sensitivity: 1.0,
            noise_scale: 0.1,
            clip_norm: 1.0,
            enabled: true,
        }
    }
}

impl DpConfig {
    /// Compute the noise standard deviation: σ = sensitivity * noise_scale.
    #[must_use]
    pub fn noise_stddev(&self) -> f64 {
        self.sensitivity * self.noise_scale
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gradient
// ─────────────────────────────────────────────────────────────────────────────

/// A flat gradient vector over a model's parameters.
#[derive(Debug, Clone)]
pub struct Gradient {
    /// Parameter deltas in the same layout as the model weights.
    pub values: Vec<f64>,
    /// Number of local samples used to compute this gradient.
    pub num_samples: usize,
}

impl Gradient {
    /// Create a zero gradient with `dim` parameters.
    #[must_use]
    pub fn zeros(dim: usize) -> Self {
        Self {
            values: vec![0.0; dim],
            num_samples: 0,
        }
    }

    /// L2 norm of the gradient values.
    #[must_use]
    pub fn l2_norm(&self) -> f64 {
        self.values.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// Clip gradient to have at most `max_norm` L2 norm.
    pub fn clip_norm(&mut self, max_norm: f64) {
        let norm = self.l2_norm();
        if norm > max_norm && norm > 0.0 {
            let scale = max_norm / norm;
            for v in &mut self.values {
                *v *= scale;
            }
        }
    }

    /// Add Gaussian noise scaled by `stddev` using a deterministic LCG seeded by `seed`.
    pub fn add_gaussian_noise(&mut self, stddev: f64, seed: u64) {
        for (i, v) in self.values.iter_mut().enumerate() {
            let noise = normal_sample(seed.wrapping_add(i as u64)) * stddev;
            *v += noise;
        }
    }

    /// Scale all gradient values by a scalar.
    pub fn scale(&mut self, factor: f64) {
        for v in &mut self.values {
            *v *= factor;
        }
    }

    /// Add another gradient in-place.
    pub fn add_assign(&mut self, other: &Self) {
        for (a, b) in self.values.iter_mut().zip(other.values.iter()) {
            *a += b;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Global model (shared embedding)
// ─────────────────────────────────────────────────────────────────────────────

/// Lightweight global item-embedding model for federated collaborative filtering.
///
/// Holds a flat vector of item embeddings (shape: `num_items × embed_dim`).
/// Gradients are indexed over the same flat layout.
#[derive(Debug, Clone)]
pub struct GlobalModel {
    /// Flat item embeddings: row-major, shape [num_items, embed_dim].
    pub weights: Vec<f64>,
    /// Number of items.
    pub num_items: usize,
    /// Embedding dimension.
    pub embed_dim: usize,
    /// Training round counter.
    pub round: u64,
}

impl GlobalModel {
    /// Create a zero-initialised global model.
    #[must_use]
    pub fn new(num_items: usize, embed_dim: usize) -> Self {
        Self {
            weights: vec![0.0; num_items * embed_dim],
            num_items,
            embed_dim,
            round: 0,
        }
    }

    /// Total number of parameters.
    #[must_use]
    pub fn param_count(&self) -> usize {
        self.weights.len()
    }

    /// Get the embedding vector for item `i` (cloned).
    ///
    /// Returns `None` if `i >= num_items`.
    #[must_use]
    pub fn item_embedding(&self, i: usize) -> Option<Vec<f64>> {
        if i >= self.num_items {
            return None;
        }
        let start = i * self.embed_dim;
        Some(self.weights[start..start + self.embed_dim].to_vec())
    }

    /// Apply a gradient update with learning rate `lr`.
    pub fn apply_gradient(&mut self, gradient: &Gradient, lr: f64) {
        for (w, g) in self.weights.iter_mut().zip(gradient.values.iter()) {
            *w -= lr * g;
        }
        self.round += 1;
    }

    /// Cosine similarity between two item embeddings.
    ///
    /// Returns `None` if either index is out of range.
    #[must_use]
    pub fn cosine_similarity(&self, i: usize, j: usize) -> Option<f64> {
        let ei = self.item_embedding(i)?;
        let ej = self.item_embedding(j)?;
        let dot: f64 = ei.iter().zip(ej.iter()).map(|(a, b)| a * b).sum();
        let ni: f64 = ei.iter().map(|a| a * a).sum::<f64>().sqrt();
        let nj: f64 = ej.iter().map(|b| b * b).sum::<f64>().sqrt();
        if ni < 1e-15 || nj < 1e-15 {
            return Some(0.0);
        }
        Some((dot / (ni * nj)).clamp(-1.0, 1.0))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Local update (uploaded by each client)
// ─────────────────────────────────────────────────────────────────────────────

/// A privacy-masked gradient uploaded by a single federated client.
#[derive(Debug, Clone)]
pub struct LocalUpdate {
    /// Client identifier (anonymous in production).
    pub client_id: String,
    /// The gradient (already clipped + noised).
    pub gradient: Gradient,
    /// Round number this update was computed for.
    pub round: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Federated client
// ─────────────────────────────────────────────────────────────────────────────

/// A single federated learning client holding private interaction data.
///
/// The client computes a local gradient via stochastic approximation of the
/// mean-squared-error loss over its known (item, rating) pairs, clips it, and
/// adds DP noise before returning a [`LocalUpdate`].
#[derive(Debug)]
pub struct FederatedClient {
    /// Client identifier.
    pub client_id: String,
    /// Private (item_index → implicit rating) map.
    interactions: HashMap<usize, f64>,
    /// DP configuration.
    dp_config: DpConfig,
    /// Local update counter (used as LCG seed diversity).
    update_count: u64,
}

impl FederatedClient {
    /// Create a new client.
    #[must_use]
    pub fn new(client_id: impl Into<String>, dp_config: DpConfig) -> Self {
        Self {
            client_id: client_id.into(),
            interactions: HashMap::new(),
            dp_config,
            update_count: 0,
        }
    }

    /// Record a user interaction (item index + implicit rating in [0, 1]).
    pub fn add_interaction(&mut self, item_idx: usize, rating: f64) {
        self.interactions.insert(item_idx, rating.clamp(0.0, 1.0));
    }

    /// Return the number of recorded interactions.
    #[must_use]
    pub fn interaction_count(&self) -> usize {
        self.interactions.len()
    }

    /// Compute a local gradient update from the current global model.
    ///
    /// The gradient is the MSE gradient averaged over local interactions:
    /// `∇L_i = -(r_ui - ê_i·e_i) * e_i`  for each item `i` the user interacted with,
    /// where `ê_i` is the predicted rating (clamped dot product).
    ///
    /// After computing the gradient:
    /// 1. Clip to `dp_config.clip_norm`.
    /// 2. Add Gaussian noise if DP is enabled.
    #[must_use]
    pub fn compute_update(&mut self, model: &GlobalModel) -> LocalUpdate {
        let mut gradient = Gradient::zeros(model.param_count());
        let n = self.interactions.len();
        if n == 0 {
            return LocalUpdate {
                client_id: self.client_id.clone(),
                gradient,
                round: model.round,
            };
        }

        // Compute MSE gradient per interaction
        for (&item_idx, &rating) in &self.interactions {
            let Some(embed) = model.item_embedding(item_idx) else {
                continue;
            };
            // Predicted rating: L2-norm-normalised dot of embedding with itself (self-score proxy)
            let norm_sq: f64 = embed.iter().map(|v| v * v).sum();
            let predicted = if norm_sq > 1e-15 { norm_sq.sqrt() } else { 0.0 };
            let error = rating - predicted.clamp(0.0, 1.0);

            // Gradient of MSE w.r.t. item embedding: -2 * error * embed / n
            let start = item_idx * model.embed_dim;
            let scale = -2.0 * error / n as f64;
            for (k, &v) in embed.iter().enumerate() {
                gradient.values[start + k] += scale * v;
            }
        }

        gradient.num_samples = n;

        // Clip
        gradient.clip_norm(self.dp_config.clip_norm);

        // Add DP noise
        if self.dp_config.enabled {
            let seed = self
                .update_count
                .wrapping_mul(2_654_435_761)
                .wrapping_add(self.client_id.len() as u64);
            gradient.add_gaussian_noise(self.dp_config.noise_stddev(), seed);
        }

        self.update_count += 1;

        LocalUpdate {
            client_id: self.client_id.clone(),
            gradient,
            round: model.round,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Federated aggregator (FedAvg)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the federated aggregator.
#[derive(Debug, Clone)]
pub struct AggregatorConfig {
    /// Learning rate for global model update.
    pub learning_rate: f64,
    /// Minimum number of client updates before aggregation.
    pub min_clients: usize,
    /// Maximum staleness (rounds) an update may have.
    pub max_staleness: u64,
}

impl Default for AggregatorConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            min_clients: 2,
            max_staleness: 5,
        }
    }
}

/// Aggregator errors.
#[derive(Debug)]
pub enum AggregatorError {
    /// Not enough fresh client updates.
    InsufficientUpdates {
        /// Number of updates available.
        available: usize,
        /// Minimum required.
        required: usize,
    },
    /// Gradient dimension mismatch.
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Received dimension.
        received: usize,
    },
}

impl std::fmt::Display for AggregatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientUpdates {
                available,
                required,
            } => write!(f, "need {required} updates, got {available}"),
            Self::DimensionMismatch { expected, received } => {
                write!(f, "gradient dim {received} != expected {expected}")
            }
        }
    }
}

/// Result alias for aggregator operations.
pub type AggregatorResult<T> = Result<T, AggregatorError>;

/// Federated aggregator implementing FedAvg.
///
/// Collects [`LocalUpdate`]s from clients and merges them into the global model
/// using sample-weighted averaging (FedAvg) once enough updates have arrived.
#[derive(Debug)]
pub struct FederatedAggregator {
    /// Current global model.
    pub model: GlobalModel,
    /// Pending client updates for the current round.
    pending: Vec<LocalUpdate>,
    /// Aggregation configuration.
    config: AggregatorConfig,
    /// Total aggregation rounds completed.
    rounds_completed: u64,
}

impl FederatedAggregator {
    /// Create a new aggregator with the given global model.
    #[must_use]
    pub fn new(model: GlobalModel, config: AggregatorConfig) -> Self {
        Self {
            model,
            pending: Vec::new(),
            config,
            rounds_completed: 0,
        }
    }

    /// Submit a local update from a client.
    ///
    /// Stale updates (older than `max_staleness` rounds) are silently dropped.
    pub fn submit_update(&mut self, update: LocalUpdate) {
        let staleness = self.model.round.saturating_sub(update.round);
        if staleness <= self.config.max_staleness {
            self.pending.push(update);
        }
    }

    /// Number of pending updates awaiting aggregation.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Attempt to aggregate pending updates via FedAvg.
    ///
    /// Returns `Ok(())` if aggregation succeeded, or an [`AggregatorError`] if
    /// there are not enough fresh updates or a gradient has wrong dimensions.
    pub fn aggregate(&mut self) -> AggregatorResult<()> {
        let n = self.pending.len();
        if n < self.config.min_clients {
            return Err(AggregatorError::InsufficientUpdates {
                available: n,
                required: self.config.min_clients,
            });
        }

        let dim = self.model.param_count();
        // Validate all gradient dims
        for upd in &self.pending {
            if upd.gradient.values.len() != dim {
                return Err(AggregatorError::DimensionMismatch {
                    expected: dim,
                    received: upd.gradient.values.len(),
                });
            }
        }

        // FedAvg: weighted average by num_samples
        let total_samples: usize = self.pending.iter().map(|u| u.gradient.num_samples).sum();
        let total_weight = if total_samples == 0 {
            n as f64
        } else {
            total_samples as f64
        };

        let mut avg_gradient = Gradient::zeros(dim);
        for upd in &self.pending {
            let weight = if total_samples == 0 {
                1.0 / n as f64
            } else {
                upd.gradient.num_samples as f64 / total_weight
            };
            for (a, b) in avg_gradient
                .values
                .iter_mut()
                .zip(upd.gradient.values.iter())
            {
                *a += weight * b;
            }
        }
        avg_gradient.num_samples = total_samples;

        // Apply to global model
        self.model
            .apply_gradient(&avg_gradient, self.config.learning_rate);
        self.pending.clear();
        self.rounds_completed += 1;
        Ok(())
    }

    /// Total number of FedAvg aggregation rounds completed.
    #[must_use]
    pub fn rounds_completed(&self) -> u64 {
        self.rounds_completed
    }

    /// Get the current aggregation configuration.
    #[must_use]
    pub fn config(&self) -> &AggregatorConfig {
        &self.config
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_model(num_items: usize, embed_dim: usize) -> GlobalModel {
        let mut m = GlobalModel::new(num_items, embed_dim);
        // Initialise with small values for non-trivial tests
        for (i, w) in m.weights.iter_mut().enumerate() {
            *w = (i as f64) * 0.01;
        }
        m
    }

    fn make_client(id: &str) -> FederatedClient {
        FederatedClient::new(
            id,
            DpConfig {
                enabled: false,
                ..Default::default()
            },
        )
    }

    // ─── DpConfig ───────────────────────────────────────────────────────────

    #[test]
    fn test_dp_config_noise_stddev() {
        let dp = DpConfig {
            sensitivity: 2.0,
            noise_scale: 0.5,
            ..Default::default()
        };
        assert!((dp.noise_stddev() - 1.0).abs() < f64::EPSILON);
    }

    // ─── Gradient ───────────────────────────────────────────────────────────

    #[test]
    fn test_gradient_zeros_and_l2() {
        let g = Gradient::zeros(4);
        assert_eq!(g.values.len(), 4);
        assert_eq!(g.l2_norm(), 0.0);
    }

    #[test]
    fn test_gradient_clip_norm() {
        let mut g = Gradient {
            values: vec![3.0, 4.0],
            num_samples: 1,
        };
        // norm = 5; clip to 1 → scale 0.2
        g.clip_norm(1.0);
        let norm_after = g.l2_norm();
        assert!((norm_after - 1.0).abs() < 1e-10, "norm = {norm_after}");
    }

    #[test]
    fn test_gradient_add_assign() {
        let mut a = Gradient {
            values: vec![1.0, 2.0],
            num_samples: 0,
        };
        let b = Gradient {
            values: vec![3.0, 4.0],
            num_samples: 0,
        };
        a.add_assign(&b);
        assert!((a.values[0] - 4.0).abs() < f64::EPSILON);
        assert!((a.values[1] - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gradient_scale() {
        let mut g = Gradient {
            values: vec![2.0, 4.0, 6.0],
            num_samples: 2,
        };
        g.scale(0.5);
        assert!((g.values[0] - 1.0).abs() < f64::EPSILON);
        assert!((g.values[2] - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gradient_gaussian_noise_changes_values() {
        let original = vec![1.0, 1.0, 1.0];
        let mut g = Gradient {
            values: original.clone(),
            num_samples: 3,
        };
        g.add_gaussian_noise(0.5, 42);
        // At least one value should have changed
        let changed = g
            .values
            .iter()
            .zip(original.iter())
            .any(|(a, b)| (a - b).abs() > 1e-12);
        assert!(changed, "noise should change at least one value");
    }

    // ─── GlobalModel ────────────────────────────────────────────────────────

    #[test]
    fn test_global_model_item_embedding() {
        let model = make_model(3, 4);
        assert!(model.item_embedding(0).is_some());
        assert!(model.item_embedding(2).is_some());
        assert!(model.item_embedding(3).is_none()); // out of range
    }

    #[test]
    fn test_global_model_cosine_similarity_same_item() {
        let model = make_model(3, 4);
        // Cosine sim of an item with itself should be 1.0 (or 0 for zero vector)
        let sim = model.cosine_similarity(1, 1);
        assert!(sim.is_some());
        let v = sim.expect("sim should exist");
        assert!(v >= 0.0, "self-similarity should be non-negative");
    }

    #[test]
    fn test_global_model_apply_gradient() {
        let mut model = make_model(2, 2);
        let initial = model.weights.clone();
        let gradient = Gradient {
            values: vec![1.0, 1.0, 1.0, 1.0],
            num_samples: 5,
        };
        model.apply_gradient(&gradient, 0.1);
        // Weights should have decreased by lr * gradient
        for (w, init) in model.weights.iter().zip(initial.iter()) {
            assert!((w - init + 0.1).abs() < 1e-10);
        }
        assert_eq!(model.round, 1);
    }

    // ─── FederatedClient ────────────────────────────────────────────────────

    #[test]
    fn test_client_add_interaction() {
        let mut client = make_client("user1");
        client.add_interaction(0, 0.8);
        client.add_interaction(2, 1.0);
        assert_eq!(client.interaction_count(), 2);
    }

    #[test]
    fn test_client_compute_update_no_interactions() {
        let mut client = make_client("user_empty");
        let model = make_model(4, 3);
        let upd = client.compute_update(&model);
        assert_eq!(upd.client_id, "user_empty");
        assert_eq!(upd.gradient.num_samples, 0);
        assert!(upd.gradient.values.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_client_compute_update_with_interactions() {
        let mut client = make_client("user2");
        client.add_interaction(0, 1.0);
        client.add_interaction(1, 0.5);
        let model = make_model(3, 4);
        let upd = client.compute_update(&model);
        assert_eq!(upd.gradient.values.len(), model.param_count());
        assert_eq!(upd.gradient.num_samples, 2);
    }

    #[test]
    fn test_client_dp_noise_applied() {
        let mut client = FederatedClient::new(
            "user_dp",
            DpConfig {
                enabled: true,
                noise_scale: 2.0, // large noise → definitely non-zero
                sensitivity: 1.0,
                clip_norm: 1.0,
            },
        );
        client.add_interaction(0, 1.0);
        let model = make_model(3, 4);
        let upd = client.compute_update(&model);
        // With large noise the gradient should have non-zero values somewhere
        let any_nonzero = upd.gradient.values.iter().any(|&v| v.abs() > 1e-12);
        assert!(any_nonzero, "DP noise should produce non-zero gradient");
    }

    // ─── FederatedAggregator ────────────────────────────────────────────────

    #[test]
    fn test_aggregator_insufficient_updates() {
        let model = make_model(3, 4);
        let config = AggregatorConfig {
            min_clients: 2,
            ..Default::default()
        };
        let mut agg = FederatedAggregator::new(model, config);
        // Submit only 1 update
        let update = LocalUpdate {
            client_id: "c1".into(),
            gradient: Gradient::zeros(3 * 4),
            round: 0,
        };
        agg.submit_update(update);
        let result = agg.aggregate();
        assert!(matches!(
            result,
            Err(AggregatorError::InsufficientUpdates { .. })
        ));
    }

    #[test]
    fn test_aggregator_fedavg_succeeds() {
        let num_items = 3;
        let embed_dim = 4;
        let model = make_model(num_items, embed_dim);
        let dim = model.param_count();
        let config = AggregatorConfig {
            min_clients: 2,
            learning_rate: 0.01,
            ..Default::default()
        };
        let mut agg = FederatedAggregator::new(model, config);

        for i in 0u64..3 {
            let g = Gradient {
                values: vec![0.1 * (i + 1) as f64; dim],
                num_samples: (i + 1) as usize,
            };
            agg.submit_update(LocalUpdate {
                client_id: format!("c{i}"),
                gradient: g,
                round: 0,
            });
        }

        assert!(agg.aggregate().is_ok());
        assert_eq!(agg.rounds_completed(), 1);
        assert_eq!(agg.pending_count(), 0);
    }

    #[test]
    fn test_aggregator_stale_update_dropped() {
        let model = make_model(2, 3);
        // Manually advance the model round
        let mut model = model;
        model.round = 10;
        let config = AggregatorConfig {
            max_staleness: 2,
            min_clients: 1,
            ..Default::default()
        };
        let mut agg = FederatedAggregator::new(model, config);
        // Update from round 5 → staleness = 5 > 2 → should be dropped
        agg.submit_update(LocalUpdate {
            client_id: "stale".into(),
            gradient: Gradient::zeros(6),
            round: 5,
        });
        assert_eq!(agg.pending_count(), 0, "stale update should be dropped");
    }

    #[test]
    fn test_full_federated_round() {
        let num_items = 5;
        let embed_dim = 4;
        let model = make_model(num_items, embed_dim);
        let config = AggregatorConfig {
            min_clients: 2,
            learning_rate: 0.05,
            ..Default::default()
        };
        let mut agg = FederatedAggregator::new(model, config);

        // Two clients submit updates
        let mut c1 = FederatedClient::new(
            "client1",
            DpConfig {
                enabled: false,
                ..Default::default()
            },
        );
        c1.add_interaction(0, 1.0);
        c1.add_interaction(2, 0.8);

        let mut c2 = FederatedClient::new(
            "client2",
            DpConfig {
                enabled: false,
                ..Default::default()
            },
        );
        c2.add_interaction(1, 0.9);
        c2.add_interaction(3, 0.7);

        let upd1 = c1.compute_update(&agg.model);
        let upd2 = c2.compute_update(&agg.model);

        agg.submit_update(upd1);
        agg.submit_update(upd2);

        assert_eq!(agg.pending_count(), 2);
        let result = agg.aggregate();
        assert!(result.is_ok(), "full round should succeed");
        assert_eq!(agg.rounds_completed(), 1);
        assert_eq!(agg.model.round, 1);
    }

    #[test]
    fn test_aggregator_dimension_mismatch() {
        let model = make_model(3, 4); // dim = 12
        let config = AggregatorConfig {
            min_clients: 1,
            ..Default::default()
        };
        let mut agg = FederatedAggregator::new(model, config);
        agg.submit_update(LocalUpdate {
            client_id: "bad".into(),
            gradient: Gradient::zeros(5), // wrong dim
            round: 0,
        });
        let result = agg.aggregate();
        assert!(matches!(
            result,
            Err(AggregatorError::DimensionMismatch { .. })
        ));
    }
}
