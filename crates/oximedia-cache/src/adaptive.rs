//! Adaptive cache threshold management.
//!
//! Dynamically adjusts cache eviction thresholds based on rolling hit-rate
//! statistics.  When the observed hit rate drops below a configurable target
//! the policy widens the cache (increases capacity or extends TTL); when the
//! hit rate exceeds the target by a comfortable margin the policy contracts
//! the cache to free resources.
//!
//! # Design
//!
//! [`AdaptivePolicy`] maintains a circular ring buffer of recent hit/miss
//! observations.  Every `adjustment_interval` operations it computes the
//! rolling hit rate over the window and compares it against the configured
//! `target_hit_rate`.
//!
//! * **Hit rate too low** → capacity is increased by `growth_factor` (capped
//!   at `max_capacity`) and TTL is extended by `ttl_extension`.
//! * **Hit rate comfortably high** → capacity is decreased by `shrink_factor`
//!   (floored at `min_capacity`) and TTL is shortened by `ttl_reduction`.
//! * **Hit rate within band** → no change.
//!
//! The policy itself does **not** own a cache; it emits [`Adjustment`]
//! recommendations that the caller applies to their cache of choice.
//!
//! # Example
//!
//! ```rust
//! use oximedia_cache::adaptive::{AdaptiveConfig, AdaptivePolicy};
//! use std::time::Duration;
//!
//! let cfg = AdaptiveConfig {
//!     target_hit_rate: 0.80,
//!     tolerance: 0.05,
//!     adjustment_interval: 20,
//!     min_capacity: 16,
//!     max_capacity: 4096,
//!     growth_factor: 1.5,
//!     shrink_factor: 0.75,
//!     ttl_extension: Duration::from_secs(10),
//!     ttl_reduction: Duration::from_secs(5),
//!     window_size: 100,
//! };
//! let mut policy = AdaptivePolicy::new(cfg).expect("valid config");
//!
//! // Simulate 20 cache hits.
//! for _ in 0..20 {
//!     let _ = policy.record_hit();
//! }
//! ```

use std::collections::VecDeque;
use std::time::Duration;

use thiserror::Error;

// ── Errors ───────────────────────────────────────────────────────────────────

/// Errors returned by [`AdaptivePolicy`] operations.
#[derive(Debug, Error)]
pub enum AdaptiveError {
    /// The supplied configuration is invalid.
    #[error("invalid adaptive config: {0}")]
    InvalidConfig(String),
}

// ── AdaptiveConfig ───────────────────────────────────────────────────────────

/// Configuration for [`AdaptivePolicy`].
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Desired cache hit rate as a fraction in `(0.0, 1.0]`.
    pub target_hit_rate: f64,
    /// Half-width of the "good enough" band around `target_hit_rate`.
    ///
    /// If the observed rate is within `[target - tolerance, target + tolerance]`
    /// no adjustment is made.
    pub tolerance: f64,
    /// Number of operations between adjustment evaluations.
    pub adjustment_interval: u64,
    /// Minimum cache capacity the policy will ever recommend.
    pub min_capacity: usize,
    /// Maximum cache capacity the policy will ever recommend.
    pub max_capacity: usize,
    /// Multiplicative factor for capacity growth (e.g. `1.5` → +50%).
    pub growth_factor: f64,
    /// Multiplicative factor for capacity shrink (e.g. `0.75` → −25%).
    pub shrink_factor: f64,
    /// Duration added to TTL when the cache is under-performing.
    pub ttl_extension: Duration,
    /// Duration removed from TTL when the cache is over-provisioned.
    pub ttl_reduction: Duration,
    /// Number of recent observations kept in the rolling window.
    pub window_size: usize,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            target_hit_rate: 0.80,
            tolerance: 0.05,
            adjustment_interval: 100,
            min_capacity: 16,
            max_capacity: 65536,
            growth_factor: 1.5,
            shrink_factor: 0.75,
            ttl_extension: Duration::from_secs(30),
            ttl_reduction: Duration::from_secs(10),
            window_size: 200,
        }
    }
}

impl AdaptiveConfig {
    /// Validate the configuration, returning an error if any field is
    /// out of range.
    pub fn validate(&self) -> Result<(), AdaptiveError> {
        if self.target_hit_rate <= 0.0 || self.target_hit_rate > 1.0 {
            return Err(AdaptiveError::InvalidConfig(
                "target_hit_rate must be in (0.0, 1.0]".into(),
            ));
        }
        if self.tolerance < 0.0 || self.tolerance >= 1.0 {
            return Err(AdaptiveError::InvalidConfig(
                "tolerance must be in [0.0, 1.0)".into(),
            ));
        }
        if self.adjustment_interval == 0 {
            return Err(AdaptiveError::InvalidConfig(
                "adjustment_interval must be > 0".into(),
            ));
        }
        if self.min_capacity == 0 {
            return Err(AdaptiveError::InvalidConfig(
                "min_capacity must be > 0".into(),
            ));
        }
        if self.max_capacity < self.min_capacity {
            return Err(AdaptiveError::InvalidConfig(
                "max_capacity must be >= min_capacity".into(),
            ));
        }
        if self.growth_factor <= 1.0 {
            return Err(AdaptiveError::InvalidConfig(
                "growth_factor must be > 1.0".into(),
            ));
        }
        if self.shrink_factor <= 0.0 || self.shrink_factor >= 1.0 {
            return Err(AdaptiveError::InvalidConfig(
                "shrink_factor must be in (0.0, 1.0)".into(),
            ));
        }
        if self.window_size == 0 {
            return Err(AdaptiveError::InvalidConfig(
                "window_size must be > 0".into(),
            ));
        }
        Ok(())
    }
}

// ── Observation ──────────────────────────────────────────────────────────────

/// A single cache operation observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Observation {
    Hit,
    Miss,
}

// ── Adjustment ───────────────────────────────────────────────────────────────

/// The direction of an adaptive adjustment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdjustmentKind {
    /// The cache should grow (hit rate is too low).
    Grow,
    /// The cache should shrink (hit rate is comfortably above target).
    Shrink,
    /// No change needed (hit rate is within tolerance band).
    NoChange,
}

/// A recommended adjustment emitted by [`AdaptivePolicy::evaluate`].
#[derive(Debug, Clone)]
pub struct Adjustment {
    /// What kind of change is recommended.
    pub kind: AdjustmentKind,
    /// Recommended new cache capacity.
    pub recommended_capacity: usize,
    /// Recommended TTL delta (positive = extend, negative = reduce).
    ///
    /// For [`AdjustmentKind::Grow`] this is `+ ttl_extension`.
    /// For [`AdjustmentKind::Shrink`] this is `- ttl_reduction` (represented
    /// as a negative duration is awkward, so we split into two fields).
    pub ttl_extend: Duration,
    /// TTL reduction (only non-zero for [`AdjustmentKind::Shrink`]).
    pub ttl_reduce: Duration,
    /// The rolling hit rate that triggered this adjustment.
    pub observed_hit_rate: f64,
}

// ── RollingWindow ────────────────────────────────────────────────────────────

/// Fixed-capacity circular buffer of observations for computing a rolling
/// hit rate.
struct RollingWindow {
    buffer: VecDeque<Observation>,
    capacity: usize,
    hits_in_window: u64,
}

impl RollingWindow {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity: capacity.max(1),
            hits_in_window: 0,
        }
    }

    fn push(&mut self, obs: Observation) {
        if self.buffer.len() == self.capacity {
            if let Some(old) = self.buffer.pop_front() {
                if old == Observation::Hit {
                    self.hits_in_window = self.hits_in_window.saturating_sub(1);
                }
            }
        }
        if obs == Observation::Hit {
            self.hits_in_window += 1;
        }
        self.buffer.push_back(obs);
    }

    fn hit_rate(&self) -> f64 {
        if self.buffer.is_empty() {
            return 0.0;
        }
        self.hits_in_window as f64 / self.buffer.len() as f64
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }

    fn clear(&mut self) {
        self.buffer.clear();
        self.hits_in_window = 0;
    }
}

// ── AdaptivePolicy ───────────────────────────────────────────────────────────

/// Adaptive cache threshold policy.
///
/// Tracks rolling hit/miss observations and periodically evaluates whether
/// the cache should grow or shrink.  The policy itself does **not** mutate
/// any cache; it produces [`Adjustment`] recommendations.
pub struct AdaptivePolicy {
    config: AdaptiveConfig,
    window: RollingWindow,
    /// Current recommended capacity (starts at the midpoint of min/max).
    current_capacity: usize,
    /// Counter of operations since the last evaluation.
    ops_since_eval: u64,
    /// Total hits recorded since creation / last reset.
    total_hits: u64,
    /// Total misses recorded since creation / last reset.
    total_misses: u64,
    /// Number of adjustments made.
    adjustments_made: u64,
    /// History of the last N adjustment snapshots.
    history: Vec<AdjustmentRecord>,
    /// Maximum history length.
    max_history: usize,
}

/// A record stored in the adjustment history.
#[derive(Debug, Clone)]
pub struct AdjustmentRecord {
    /// The adjustment that was made.
    pub adjustment: Adjustment,
    /// Cumulative operation count at the time of this adjustment.
    pub at_operation: u64,
}

impl AdaptivePolicy {
    /// Create a new policy from the given configuration.
    ///
    /// Returns an error if the configuration is invalid.
    pub fn new(config: AdaptiveConfig) -> Result<AdaptivePolicy, AdaptiveError> {
        config.validate()?;
        let initial_capacity = config.min_capacity
            + (config.max_capacity - config.min_capacity) / 2;
        let window = RollingWindow::new(config.window_size);
        Ok(Self {
            config,
            window,
            current_capacity: initial_capacity,
            ops_since_eval: 0,
            total_hits: 0,
            total_misses: 0,
            adjustments_made: 0,
            history: Vec::new(),
            max_history: 64,
        })
    }

    /// Create a policy with a specific initial capacity.
    pub fn with_initial_capacity(
        config: AdaptiveConfig,
        initial_capacity: usize,
    ) -> Result<AdaptivePolicy, AdaptiveError> {
        config.validate()?;
        let clamped = initial_capacity
            .max(config.min_capacity)
            .min(config.max_capacity);
        let window = RollingWindow::new(config.window_size);
        Ok(Self {
            config,
            window,
            current_capacity: clamped,
            ops_since_eval: 0,
            total_hits: 0,
            total_misses: 0,
            adjustments_made: 0,
            history: Vec::new(),
            max_history: 64,
        })
    }

    // ── Recording ────────────────────────────────────────────────────────────

    /// Record a cache hit.
    ///
    /// Returns `Some(Adjustment)` when this observation triggers an
    /// evaluation (every `adjustment_interval` operations), `None` otherwise.
    pub fn record_hit(&mut self) -> Option<Adjustment> {
        self.total_hits += 1;
        self.window.push(Observation::Hit);
        self.ops_since_eval += 1;
        self.maybe_evaluate()
    }

    /// Record a cache miss.
    ///
    /// Returns `Some(Adjustment)` when this observation triggers an
    /// evaluation, `None` otherwise.
    pub fn record_miss(&mut self) -> Option<Adjustment> {
        self.total_misses += 1;
        self.window.push(Observation::Miss);
        self.ops_since_eval += 1;
        self.maybe_evaluate()
    }

    /// Force an evaluation regardless of the operation counter.
    pub fn evaluate_now(&mut self) -> Adjustment {
        self.ops_since_eval = 0;
        self.compute_adjustment()
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    /// Current rolling hit rate.
    pub fn rolling_hit_rate(&self) -> f64 {
        self.window.hit_rate()
    }

    /// Total hits since creation / last reset.
    pub fn total_hits(&self) -> u64 {
        self.total_hits
    }

    /// Total misses since creation / last reset.
    pub fn total_misses(&self) -> u64 {
        self.total_misses
    }

    /// Lifetime hit rate (not rolling-windowed).
    pub fn lifetime_hit_rate(&self) -> f64 {
        let total = self.total_hits + self.total_misses;
        if total == 0 {
            0.0
        } else {
            self.total_hits as f64 / total as f64
        }
    }

    /// Currently recommended capacity.
    pub fn current_capacity(&self) -> usize {
        self.current_capacity
    }

    /// Number of adjustments made so far.
    pub fn adjustments_made(&self) -> u64 {
        self.adjustments_made
    }

    /// Read-only access to the adjustment history.
    pub fn history(&self) -> &[AdjustmentRecord] {
        &self.history
    }

    /// The active configuration.
    pub fn config(&self) -> &AdaptiveConfig {
        &self.config
    }

    /// Total number of observations in the rolling window.
    pub fn window_fill(&self) -> usize {
        self.window.len()
    }

    // ── Reset ────────────────────────────────────────────────────────────────

    /// Reset all counters and the rolling window, keeping the current
    /// configuration.
    pub fn reset(&mut self) {
        self.window.clear();
        self.ops_since_eval = 0;
        self.total_hits = 0;
        self.total_misses = 0;
        self.adjustments_made = 0;
        self.history.clear();
        self.current_capacity = self.config.min_capacity
            + (self.config.max_capacity - self.config.min_capacity) / 2;
    }

    // ── Internals ────────────────────────────────────────────────────────────

    fn maybe_evaluate(&mut self) -> Option<Adjustment> {
        if self.ops_since_eval >= self.config.adjustment_interval {
            self.ops_since_eval = 0;
            let adj = self.compute_adjustment();
            Some(adj)
        } else {
            None
        }
    }

    fn compute_adjustment(&mut self) -> Adjustment {
        let rate = self.window.hit_rate();
        let target = self.config.target_hit_rate;
        let tol = self.config.tolerance;

        let (kind, new_cap, ttl_ext, ttl_red) = if rate < target - tol {
            // Under-performing → grow.
            let raw = (self.current_capacity as f64 * self.config.growth_factor).ceil() as usize;
            let capped = raw.min(self.config.max_capacity);
            (
                AdjustmentKind::Grow,
                capped,
                self.config.ttl_extension,
                Duration::ZERO,
            )
        } else if rate > target + tol {
            // Over-provisioned → shrink.
            let raw = (self.current_capacity as f64 * self.config.shrink_factor).floor() as usize;
            let floored = raw.max(self.config.min_capacity);
            (
                AdjustmentKind::Shrink,
                floored,
                Duration::ZERO,
                self.config.ttl_reduction,
            )
        } else {
            // Within tolerance band → no change.
            (
                AdjustmentKind::NoChange,
                self.current_capacity,
                Duration::ZERO,
                Duration::ZERO,
            )
        };

        self.current_capacity = new_cap;
        if kind != AdjustmentKind::NoChange {
            self.adjustments_made += 1;
        }

        let adj = Adjustment {
            kind,
            recommended_capacity: new_cap,
            ttl_extend: ttl_ext,
            ttl_reduce: ttl_red,
            observed_hit_rate: rate,
        };

        // Record in history, capping at max_history.
        let total_ops = self.total_hits + self.total_misses;
        self.history.push(AdjustmentRecord {
            adjustment: adj.clone(),
            at_operation: total_ops,
        });
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }

        adj
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> AdaptiveConfig {
        AdaptiveConfig {
            target_hit_rate: 0.80,
            tolerance: 0.05,
            adjustment_interval: 10,
            min_capacity: 8,
            max_capacity: 256,
            growth_factor: 2.0,
            shrink_factor: 0.5,
            ttl_extension: Duration::from_secs(10),
            ttl_reduction: Duration::from_secs(5),
            window_size: 50,
        }
    }

    // 1. Valid config creates policy successfully
    #[test]
    fn test_valid_config_creates_policy() {
        let cfg = default_config();
        let policy = AdaptivePolicy::new(cfg);
        assert!(policy.is_ok());
    }

    // 2. Invalid target_hit_rate is rejected
    #[test]
    fn test_invalid_target_hit_rate() {
        let mut cfg = default_config();
        cfg.target_hit_rate = 0.0;
        assert!(AdaptivePolicy::new(cfg.clone()).is_err());

        cfg.target_hit_rate = 1.5;
        assert!(AdaptivePolicy::new(cfg).is_err());
    }

    // 3. Invalid growth_factor is rejected
    #[test]
    fn test_invalid_growth_factor() {
        let mut cfg = default_config();
        cfg.growth_factor = 0.5;
        assert!(AdaptivePolicy::new(cfg).is_err());
    }

    // 4. Invalid shrink_factor is rejected
    #[test]
    fn test_invalid_shrink_factor() {
        let mut cfg = default_config();
        cfg.shrink_factor = 1.0;
        assert!(AdaptivePolicy::new(cfg.clone()).is_err());

        cfg.shrink_factor = 0.0;
        assert!(AdaptivePolicy::new(cfg).is_err());
    }

    // 5. All hits → capacity shrinks (hit rate above target + tolerance)
    #[test]
    fn test_all_hits_triggers_shrink() {
        let cfg = default_config();
        let mut policy = AdaptivePolicy::new(cfg).expect("valid config");
        let initial_cap = policy.current_capacity();

        // Record 10 hits → triggers evaluation
        let mut adj = None;
        for _ in 0..10 {
            adj = policy.record_hit();
        }
        let adjustment = adj.expect("should trigger after 10 ops");
        assert_eq!(adjustment.kind, AdjustmentKind::Shrink);
        assert!(
            policy.current_capacity() < initial_cap,
            "capacity should have decreased"
        );
    }

    // 6. All misses → capacity grows (hit rate below target - tolerance)
    #[test]
    fn test_all_misses_triggers_grow() {
        let cfg = default_config();
        let mut policy = AdaptivePolicy::new(cfg).expect("valid config");
        let initial_cap = policy.current_capacity();

        let mut adj = None;
        for _ in 0..10 {
            adj = policy.record_miss();
        }
        let adjustment = adj.expect("should trigger after 10 ops");
        assert_eq!(adjustment.kind, AdjustmentKind::Grow);
        assert!(
            policy.current_capacity() > initial_cap,
            "capacity should have increased"
        );
    }

    // 7. Hit rate within tolerance band → no change
    #[test]
    fn test_within_tolerance_no_change() {
        let cfg = default_config(); // target 0.80, tolerance 0.05
        let mut policy = AdaptivePolicy::new(cfg).expect("valid config");

        // 8 hits + 2 misses = 0.80 hit rate → within [0.75, 0.85]
        for _ in 0..8 {
            let _ = policy.record_hit();
        }
        let mut adj = None;
        for _ in 0..2 {
            adj = policy.record_miss();
        }
        let adjustment = adj.expect("should trigger after 10 ops");
        assert_eq!(adjustment.kind, AdjustmentKind::NoChange);
    }

    // 8. Capacity never exceeds max_capacity
    #[test]
    fn test_capacity_capped_at_max() {
        let mut cfg = default_config();
        cfg.max_capacity = 300;
        cfg.growth_factor = 100.0; // aggressive growth
        let mut policy = AdaptivePolicy::new(cfg).expect("valid config");

        // All misses → grows aggressively
        for _ in 0..10 {
            let _ = policy.record_miss();
        }
        assert!(
            policy.current_capacity() <= 300,
            "capacity {} should not exceed max 300",
            policy.current_capacity()
        );
    }

    // 9. Capacity never drops below min_capacity
    #[test]
    fn test_capacity_floored_at_min() {
        let mut cfg = default_config();
        cfg.min_capacity = 4;
        cfg.shrink_factor = 0.01; // aggressive shrink
        let mut policy = AdaptivePolicy::new(cfg).expect("valid config");

        // All hits → shrinks aggressively
        for _ in 0..10 {
            let _ = policy.record_hit();
        }
        assert!(
            policy.current_capacity() >= 4,
            "capacity {} should not drop below min 4",
            policy.current_capacity()
        );
    }

    // 10. Rolling window only considers recent observations
    #[test]
    fn test_rolling_window_discards_old() {
        let mut cfg = default_config();
        cfg.window_size = 10;
        cfg.adjustment_interval = 10;
        let mut policy = AdaptivePolicy::new(cfg).expect("valid config");

        // Fill window with misses (hit rate = 0)
        for _ in 0..10 {
            let _ = policy.record_miss();
        }
        // Now fill window with hits (old misses get evicted)
        for _ in 0..10 {
            let _ = policy.record_hit();
        }
        // Rolling window should now be all hits → hit rate = 1.0
        let rate = policy.rolling_hit_rate();
        assert!(
            (rate - 1.0).abs() < 1e-9,
            "rolling hit rate should be 1.0, got {rate}"
        );
    }

    // 11. evaluate_now forces an immediate evaluation
    #[test]
    fn test_evaluate_now() {
        let cfg = default_config();
        let mut policy = AdaptivePolicy::new(cfg).expect("valid config");

        // Record fewer ops than adjustment_interval
        for _ in 0..3 {
            let _ = policy.record_hit();
        }
        // Force evaluation
        let adj = policy.evaluate_now();
        // With only 3 hits → hit rate 1.0 → shrink expected
        assert_eq!(adj.kind, AdjustmentKind::Shrink);
    }

    // 12. reset clears all state
    #[test]
    fn test_reset_clears_state() {
        let cfg = default_config();
        let mut policy = AdaptivePolicy::new(cfg).expect("valid config");

        for _ in 0..20 {
            let _ = policy.record_hit();
        }
        assert!(policy.total_hits() > 0);
        assert!(!policy.history().is_empty());

        policy.reset();
        assert_eq!(policy.total_hits(), 0);
        assert_eq!(policy.total_misses(), 0);
        assert_eq!(policy.adjustments_made(), 0);
        assert!(policy.history().is_empty());
        assert_eq!(policy.window_fill(), 0);
    }

    // 13. TTL extension is set on grow
    #[test]
    fn test_ttl_extension_on_grow() {
        let cfg = default_config();
        let mut policy = AdaptivePolicy::new(cfg).expect("valid config");

        let mut adj = None;
        for _ in 0..10 {
            adj = policy.record_miss();
        }
        let adjustment = adj.expect("should trigger");
        assert_eq!(adjustment.kind, AdjustmentKind::Grow);
        assert_eq!(adjustment.ttl_extend, Duration::from_secs(10));
        assert_eq!(adjustment.ttl_reduce, Duration::ZERO);
    }

    // 14. TTL reduction is set on shrink
    #[test]
    fn test_ttl_reduction_on_shrink() {
        let cfg = default_config();
        let mut policy = AdaptivePolicy::new(cfg).expect("valid config");

        let mut adj = None;
        for _ in 0..10 {
            adj = policy.record_hit();
        }
        let adjustment = adj.expect("should trigger");
        assert_eq!(adjustment.kind, AdjustmentKind::Shrink);
        assert_eq!(adjustment.ttl_extend, Duration::ZERO);
        assert_eq!(adjustment.ttl_reduce, Duration::from_secs(5));
    }

    // 15. lifetime_hit_rate covers all observations
    #[test]
    fn test_lifetime_hit_rate() {
        let cfg = default_config();
        let mut policy = AdaptivePolicy::new(cfg).expect("valid config");

        for _ in 0..7 {
            let _ = policy.record_hit();
        }
        for _ in 0..3 {
            let _ = policy.record_miss();
        }
        let rate = policy.lifetime_hit_rate();
        assert!(
            (rate - 0.7).abs() < 1e-9,
            "expected 0.7, got {rate}"
        );
    }

    // 16. adjustment history records entries
    #[test]
    fn test_history_records() {
        let cfg = default_config(); // interval = 10
        let mut policy = AdaptivePolicy::new(cfg).expect("valid config");

        // Trigger two evaluations
        for _ in 0..20 {
            let _ = policy.record_hit();
        }
        assert!(
            policy.history().len() >= 2,
            "expected at least 2 history records, got {}",
            policy.history().len()
        );
    }

    // 17. with_initial_capacity clamps to min/max
    #[test]
    fn test_with_initial_capacity_clamps() {
        let cfg = default_config(); // min=8, max=256
        let p1 = AdaptivePolicy::with_initial_capacity(cfg.clone(), 2)
            .expect("valid");
        assert_eq!(p1.current_capacity(), 8, "should clamp to min");

        let p2 = AdaptivePolicy::with_initial_capacity(cfg, 9999)
            .expect("valid");
        assert_eq!(p2.current_capacity(), 256, "should clamp to max");
    }

    // 18. max_capacity == min_capacity is valid
    #[test]
    fn test_min_equals_max_capacity() {
        let mut cfg = default_config();
        cfg.min_capacity = 64;
        cfg.max_capacity = 64;
        let mut policy = AdaptivePolicy::new(cfg).expect("valid config");

        // Even with all misses, capacity cannot grow beyond 64
        for _ in 0..10 {
            let _ = policy.record_miss();
        }
        assert_eq!(policy.current_capacity(), 64);

        // Even with all hits, capacity cannot shrink below 64
        for _ in 0..10 {
            let _ = policy.record_hit();
        }
        assert_eq!(policy.current_capacity(), 64);
    }

    // 19. adjustment_interval 0 is rejected
    #[test]
    fn test_zero_adjustment_interval_rejected() {
        let mut cfg = default_config();
        cfg.adjustment_interval = 0;
        assert!(AdaptivePolicy::new(cfg).is_err());
    }

    // 20. window_size 0 is rejected
    #[test]
    fn test_zero_window_size_rejected() {
        let mut cfg = default_config();
        cfg.window_size = 0;
        assert!(AdaptivePolicy::new(cfg).is_err());
    }

    // 21. Successive grow adjustments increase capacity monotonically
    #[test]
    fn test_successive_grow_monotonic() {
        let mut cfg = default_config();
        cfg.max_capacity = 100_000;
        let mut policy = AdaptivePolicy::new(cfg).expect("valid config");

        let mut prev_cap = policy.current_capacity();
        for round in 0..5 {
            for _ in 0..10 {
                let _ = policy.record_miss();
            }
            let cap = policy.current_capacity();
            assert!(
                cap >= prev_cap,
                "round {round}: capacity should not decrease on grow ({prev_cap} → {cap})"
            );
            prev_cap = cap;
        }
    }

    // 22. observed_hit_rate is populated in the adjustment
    #[test]
    fn test_observed_hit_rate_in_adjustment() {
        let cfg = default_config();
        let mut policy = AdaptivePolicy::new(cfg).expect("valid config");

        for _ in 0..5 {
            let _ = policy.record_hit();
        }
        for _ in 0..5 {
            let _ = policy.record_miss();
        }
        // Last record_miss triggers eval at op 10
        let last_hist = policy.history().last().expect("should have history");
        let rate = last_hist.adjustment.observed_hit_rate;
        assert!(
            (rate - 0.5).abs() < 1e-9,
            "expected 0.5 hit rate, got {rate}"
        );
    }
}
