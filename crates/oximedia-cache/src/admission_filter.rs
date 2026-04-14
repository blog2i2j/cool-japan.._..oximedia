//! Adaptive tier promotion thresholds based on access frequency.
//!
//! In a multi-tier cache, moving (promoting) an entry from a slower tier (L2)
//! to a faster tier (L1) costs bandwidth and L1 capacity.  Naïve policies
//! promote on every re-access, causing *scan pollution* where short-lived
//! sequential reads evict long-lived hot entries.
//!
//! [`AdmissionFilter`] implements a frequency-gated admission policy:
//!
//! 1. Each key's access count is maintained with exponential decay to favour
//!    *recently* frequent keys over historically frequent ones.
//! 2. A key is admitted to the hot tier only when its decayed frequency
//!    exceeds a configurable `admission_threshold`.
//! 3. The threshold is adapted dynamically: when the hot-tier hit rate
//!    exceeds `target_hit_rate` the threshold is relaxed; when it falls below
//!    the threshold is tightened.
//!
//! # Example
//!
//! ```rust
//! use oximedia_cache::admission_filter::{AdmissionFilter, AdmissionConfig};
//!
//! let cfg = AdmissionConfig {
//!     admission_threshold: 3.0,
//!     target_hit_rate: 0.80,
//!     decay_factor: 0.90,
//!     adjust_interval: 10,
//!     threshold_step: 0.5,
//!     min_threshold: 1.0,
//!     max_threshold: 20.0,
//! };
//! let mut filter = AdmissionFilter::new(cfg).expect("valid config");
//!
//! // Record accesses to "hot-key"
//! for _ in 0..5 {
//!     filter.record_access("hot-key");
//! }
//! assert!(filter.should_admit("hot-key"));
//! assert!(!filter.should_admit("cold-key"));
//! ```

use std::collections::HashMap;

use thiserror::Error;

// ── Errors ────────────────────────────────────────────────────────────────────

/// Errors produced by [`AdmissionFilter`] operations.
#[derive(Debug, Error)]
pub enum AdmissionError {
    /// The supplied [`AdmissionConfig`] has an invalid field value.
    #[error("invalid admission config: {0}")]
    InvalidConfig(String),
}

// ── AdmissionConfig ───────────────────────────────────────────────────────────

/// Configuration for [`AdmissionFilter`].
#[derive(Debug, Clone)]
pub struct AdmissionConfig {
    /// Access-frequency score a key must reach before being admitted to the
    /// hot tier.  Must be ≥ 1.0.
    pub admission_threshold: f64,
    /// Target hot-tier hit rate.  The filter tightens the threshold when the
    /// observed rate falls below this value and relaxes it when above.
    /// Must be in `(0.0, 1.0]`.
    pub target_hit_rate: f64,
    /// Per-access decay applied to *all* counters: `counter *= decay_factor`.
    /// Must be in `(0.0, 1.0]`.  Values closer to 1.0 give longer memory;
    /// values closer to 0.0 favour only the most recent accesses.
    pub decay_factor: f64,
    /// Number of admission decisions between threshold adjustments.
    pub adjust_interval: u64,
    /// Amount the threshold is raised or lowered per adjustment.
    pub threshold_step: f64,
    /// Minimum threshold (lower bound after relaxation).  Must be ≥ 1.0.
    pub min_threshold: f64,
    /// Maximum threshold (upper bound after tightening).
    pub max_threshold: f64,
}

impl Default for AdmissionConfig {
    fn default() -> Self {
        Self {
            admission_threshold: 5.0,
            target_hit_rate: 0.80,
            decay_factor: 0.95,
            adjust_interval: 100,
            threshold_step: 0.5,
            min_threshold: 1.0,
            max_threshold: 50.0,
        }
    }
}

impl AdmissionConfig {
    /// Validate the configuration, returning an error if any field is invalid.
    pub fn validate(&self) -> Result<(), AdmissionError> {
        if self.admission_threshold < 1.0 {
            return Err(AdmissionError::InvalidConfig(
                "admission_threshold must be >= 1.0".into(),
            ));
        }
        if self.target_hit_rate <= 0.0 || self.target_hit_rate > 1.0 {
            return Err(AdmissionError::InvalidConfig(
                "target_hit_rate must be in (0.0, 1.0]".into(),
            ));
        }
        if self.decay_factor <= 0.0 || self.decay_factor > 1.0 {
            return Err(AdmissionError::InvalidConfig(
                "decay_factor must be in (0.0, 1.0]".into(),
            ));
        }
        if self.adjust_interval == 0 {
            return Err(AdmissionError::InvalidConfig(
                "adjust_interval must be > 0".into(),
            ));
        }
        if self.min_threshold < 1.0 {
            return Err(AdmissionError::InvalidConfig(
                "min_threshold must be >= 1.0".into(),
            ));
        }
        if self.max_threshold < self.min_threshold {
            return Err(AdmissionError::InvalidConfig(
                "max_threshold must be >= min_threshold".into(),
            ));
        }
        Ok(())
    }
}

// ── AdmissionDecision ─────────────────────────────────────────────────────────

/// The outcome of a [`AdmissionFilter::admit`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionDecision {
    /// The key has earned sufficient frequency to be promoted.
    Admit,
    /// The key is not yet hot enough.
    Deny,
}

// ── AdmissionFilter ───────────────────────────────────────────────────────────

/// Frequency-gated tier-promotion admission filter with adaptive threshold.
pub struct AdmissionFilter {
    config: AdmissionConfig,
    /// Decayed frequency counters per key.
    counters: HashMap<String, f64>,
    /// Current admission threshold (adapts over time).
    threshold: f64,
    /// Number of decisions since the last threshold adjustment.
    decisions_since_adjust: u64,
    /// Hot-tier hits recorded since last adjustment.
    hot_hits: u64,
    /// Hot-tier misses recorded since last adjustment.
    hot_misses: u64,
    /// Lifetime total of access events processed.
    total_accesses: u64,
    /// Lifetime total of `Admit` decisions.
    total_admits: u64,
    /// Lifetime total of `Deny` decisions.
    total_denies: u64,
}

impl AdmissionFilter {
    /// Create a new `AdmissionFilter` from the given configuration.
    ///
    /// Returns an error if the configuration is invalid.
    pub fn new(config: AdmissionConfig) -> Result<Self, AdmissionError> {
        config.validate()?;
        let threshold = config.admission_threshold;
        Ok(Self {
            config,
            counters: HashMap::new(),
            threshold,
            decisions_since_adjust: 0,
            hot_hits: 0,
            hot_misses: 0,
            total_accesses: 0,
            total_admits: 0,
            total_denies: 0,
        })
    }

    // ── Access recording ─────────────────────────────────────────────────────

    /// Record one access to `key`.
    ///
    /// Applies exponential decay to all tracked counters and increments the
    /// counter for `key` by 1.0.
    pub fn record_access(&mut self, key: &str) {
        self.total_accesses += 1;

        // Decay all counters.
        let factor = self.config.decay_factor;
        for v in self.counters.values_mut() {
            *v *= factor;
        }

        // Increment this key's counter.
        let counter = self.counters.entry(key.to_string()).or_insert(0.0);
        *counter += 1.0;

        // Prune counters that have decayed below a negligible threshold so
        // the map does not grow without bound.
        self.counters.retain(|_, v| *v > 0.001);
    }

    /// Return `true` when `key` should be promoted to the hot tier.
    ///
    /// Equivalent to calling [`admit`] and checking whether the decision is
    /// [`AdmissionDecision::Admit`].
    #[must_use]
    pub fn should_admit(&self, key: &str) -> bool {
        let freq = self.counters.get(key).copied().unwrap_or(0.0);
        freq >= self.threshold
    }

    /// Make an explicit admission decision for `key`, update counters, and
    /// potentially trigger a threshold adjustment.
    pub fn admit(&mut self, key: &str) -> AdmissionDecision {
        let freq = self.counters.get(key).copied().unwrap_or(0.0);
        let decision = if freq >= self.threshold {
            self.total_admits += 1;
            AdmissionDecision::Admit
        } else {
            self.total_denies += 1;
            AdmissionDecision::Deny
        };

        self.decisions_since_adjust += 1;
        if self.decisions_since_adjust >= self.config.adjust_interval {
            self.adjust_threshold();
            self.decisions_since_adjust = 0;
            self.hot_hits = 0;
            self.hot_misses = 0;
        }

        decision
    }

    /// Inform the filter that the hot tier produced a hit.
    ///
    /// Used by the adaptive threshold adjuster to measure hot-tier effectiveness.
    pub fn record_hot_hit(&mut self) {
        self.hot_hits += 1;
    }

    /// Inform the filter that the hot tier produced a miss.
    pub fn record_hot_miss(&mut self) {
        self.hot_misses += 1;
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    /// The current admission threshold.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// The decayed access frequency for `key` (0.0 if never seen).
    #[must_use]
    pub fn frequency(&self, key: &str) -> f64 {
        self.counters.get(key).copied().unwrap_or(0.0)
    }

    /// Total number of `Admit` decisions made.
    #[must_use]
    pub fn total_admits(&self) -> u64 {
        self.total_admits
    }

    /// Total number of `Deny` decisions made.
    #[must_use]
    pub fn total_denies(&self) -> u64 {
        self.total_denies
    }

    /// Total access events processed.
    #[must_use]
    pub fn total_accesses(&self) -> u64 {
        self.total_accesses
    }

    /// Number of distinct keys currently tracked (with non-negligible frequency).
    #[must_use]
    pub fn tracked_keys(&self) -> usize {
        self.counters.len()
    }

    /// The hit rate observed in the hot tier since the last threshold adjustment.
    #[must_use]
    pub fn hot_tier_hit_rate(&self) -> f64 {
        let total = self.hot_hits + self.hot_misses;
        if total == 0 {
            0.0
        } else {
            self.hot_hits as f64 / total as f64
        }
    }

    /// The active configuration.
    #[must_use]
    pub fn config(&self) -> &AdmissionConfig {
        &self.config
    }

    // ── Reset ────────────────────────────────────────────────────────────────

    /// Reset all counters and the threshold back to the configured initial value.
    pub fn reset(&mut self) {
        self.counters.clear();
        self.threshold = self.config.admission_threshold;
        self.decisions_since_adjust = 0;
        self.hot_hits = 0;
        self.hot_misses = 0;
        self.total_accesses = 0;
        self.total_admits = 0;
        self.total_denies = 0;
    }

    // ── Internal ─────────────────────────────────────────────────────────────

    /// Adjust the admission threshold based on the observed hot-tier hit rate.
    fn adjust_threshold(&mut self) {
        let rate = self.hot_tier_hit_rate();
        let step = self.config.threshold_step;

        if rate < self.config.target_hit_rate {
            // Hit rate too low → raise threshold to admit fewer, higher-quality
            // entries (be more selective).
            self.threshold = (self.threshold + step).min(self.config.max_threshold);
        } else if rate > self.config.target_hit_rate {
            // Hit rate above target → relax threshold to admit more entries.
            self.threshold = (self.threshold - step).max(self.config.min_threshold);
        }
        // If exactly on target: no change.
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_filter() -> AdmissionFilter {
        let cfg = AdmissionConfig {
            admission_threshold: 3.0,
            target_hit_rate: 0.80,
            decay_factor: 0.90,
            adjust_interval: 10,
            threshold_step: 0.5,
            min_threshold: 1.0,
            max_threshold: 20.0,
        };
        AdmissionFilter::new(cfg).expect("valid config")
    }

    // 1. New filter: no keys admitted
    #[test]
    fn test_new_filter_admits_nothing() {
        let filter = default_filter();
        assert!(!filter.should_admit("any-key"));
        assert_eq!(filter.tracked_keys(), 0);
    }

    // 2. After sufficient accesses a key is admitted
    #[test]
    fn test_key_admitted_after_accesses() {
        let mut filter = default_filter(); // threshold = 3.0
        for _ in 0..5 {
            filter.record_access("hot");
        }
        assert!(filter.should_admit("hot"), "hot-key should be admitted");
    }

    // 3. Cold key is not admitted
    #[test]
    fn test_cold_key_denied() {
        let mut filter = default_filter();
        filter.record_access("cold"); // only 1 access
        assert!(!filter.should_admit("cold"));
    }

    // 4. admit() returns Admit for a hot key
    #[test]
    fn test_admit_decision_admit() {
        let mut filter = default_filter();
        for _ in 0..5 {
            filter.record_access("k");
        }
        assert_eq!(filter.admit("k"), AdmissionDecision::Admit);
    }

    // 5. admit() returns Deny for a cold key
    #[test]
    fn test_admit_decision_deny() {
        let mut filter = default_filter();
        assert_eq!(filter.admit("unknown"), AdmissionDecision::Deny);
    }

    // 6. Decay reduces frequency over time
    #[test]
    fn test_decay_reduces_frequency() {
        let mut filter = default_filter(); // decay = 0.90
        // 5 accesses → freq ≈ 5 (before decay stabilises)
        for _ in 0..5 {
            filter.record_access("k");
        }
        let freq_after = filter.frequency("k");
        // Record 100 more accesses to OTHER keys to trigger decay without
        // touching "k".
        for i in 0..50 {
            filter.record_access(&format!("other-{i}"));
        }
        let freq_decayed = filter.frequency("k");
        assert!(
            freq_decayed < freq_after,
            "frequency should decay: {freq_after} → {freq_decayed}"
        );
    }

    // 7. total_accesses counter increments
    #[test]
    fn test_total_accesses_counter() {
        let mut filter = default_filter();
        for _ in 0..7 {
            filter.record_access("k");
        }
        assert_eq!(filter.total_accesses(), 7);
    }

    // 8. total_admits / total_denies counters
    #[test]
    fn test_admits_denies_counters() {
        let mut filter = default_filter();
        for _ in 0..5 {
            filter.record_access("hot");
        }
        let _ = filter.admit("hot");      // Admit
        let _ = filter.admit("cold");     // Deny
        assert_eq!(filter.total_admits(), 1);
        assert_eq!(filter.total_denies(), 1);
    }

    // 9. threshold tightens when hot-tier hit rate is too low
    #[test]
    fn test_threshold_tightens_on_low_hit_rate() {
        let mut filter = default_filter(); // step = 0.5, target = 0.80
        let initial_threshold = filter.threshold();

        // Record many hot misses (zero hits → rate = 0 < 0.80).
        for _ in 0..10 {
            filter.record_hot_miss();
        }
        // Trigger threshold adjustment by making adjust_interval decisions.
        for _ in 0..10 {
            let _ = filter.admit("x");
        }
        // Threshold should have increased.
        assert!(
            filter.threshold() > initial_threshold,
            "threshold should tighten: {} → {}",
            initial_threshold,
            filter.threshold()
        );
    }

    // 10. threshold relaxes when hot-tier hit rate exceeds target
    #[test]
    fn test_threshold_relaxes_on_high_hit_rate() {
        let mut filter = default_filter();
        let initial_threshold = filter.threshold();

        // Record many hot hits (no misses → rate = 1.0 > 0.80).
        for _ in 0..20 {
            filter.record_hot_hit();
        }
        // Trigger adjustment.
        for _ in 0..10 {
            let _ = filter.admit("x");
        }
        assert!(
            filter.threshold() < initial_threshold,
            "threshold should relax: {} → {}",
            initial_threshold,
            filter.threshold()
        );
    }

    // 11. threshold never exceeds max_threshold
    #[test]
    fn test_threshold_capped_at_max() {
        let cfg = AdmissionConfig {
            admission_threshold: 19.5,
            max_threshold: 20.0,
            threshold_step: 1.0,
            adjust_interval: 5,
            ..AdmissionConfig::default()
        };
        let mut filter = AdmissionFilter::new(cfg).expect("valid");

        for _ in 0..5 {
            filter.record_hot_miss();
        }
        for _ in 0..5 {
            let _ = filter.admit("x");
        }
        assert!(filter.threshold() <= 20.0);
    }

    // 12. threshold never drops below min_threshold
    #[test]
    fn test_threshold_floored_at_min() {
        let cfg = AdmissionConfig {
            admission_threshold: 1.5,
            min_threshold: 1.0,
            threshold_step: 1.0,
            adjust_interval: 5,
            ..AdmissionConfig::default()
        };
        let mut filter = AdmissionFilter::new(cfg).expect("valid");

        for _ in 0..10 {
            filter.record_hot_hit();
        }
        for _ in 0..5 {
            let _ = filter.admit("x");
        }
        assert!(filter.threshold() >= 1.0);
    }

    // 13. reset() clears all state
    #[test]
    fn test_reset_clears_state() {
        let mut filter = default_filter();
        for _ in 0..5 {
            filter.record_access("k");
        }
        let _ = filter.admit("k");
        filter.reset();
        assert_eq!(filter.total_accesses(), 0);
        assert_eq!(filter.total_admits(), 0);
        assert_eq!(filter.total_denies(), 0);
        assert_eq!(filter.tracked_keys(), 0);
        assert_eq!(filter.threshold(), 3.0); // back to initial config value
        assert!(!filter.should_admit("k"));
    }

    // 14. invalid config is rejected
    #[test]
    fn test_invalid_config_rejected() {
        let mut cfg = AdmissionConfig::default();
        cfg.admission_threshold = 0.5; // < 1.0
        assert!(AdmissionFilter::new(cfg).is_err());
    }

    // 15. hot_tier_hit_rate is 0 when no feedback recorded
    #[test]
    fn test_hot_tier_hit_rate_zero_initially() {
        let filter = default_filter();
        assert_eq!(filter.hit_rate_zero_check(), 0.0);
    }

    // 16. tracked_keys drops after decay prunes negligible entries
    #[test]
    fn test_tracked_keys_pruned_after_decay() {
        let cfg = AdmissionConfig {
            decay_factor: 0.01, // aggressive decay
            admission_threshold: 3.0,
            adjust_interval: 1000,
            ..AdmissionConfig::default()
        };
        let mut filter = AdmissionFilter::new(cfg).expect("valid");
        filter.record_access("ephemeral");
        let before = filter.tracked_keys();
        // Trigger decay 100 times by accessing another key repeatedly.
        for _ in 0..100 {
            filter.record_access("other");
        }
        let after = filter.tracked_keys();
        // "ephemeral" should have decayed below 0.001 and been pruned.
        assert!(
            after <= before,
            "tracked keys should not grow: {before} → {after}"
        );
    }

    // 17. frequency() returns 0 for unseen key
    #[test]
    fn test_frequency_unknown_key() {
        let filter = default_filter();
        assert_eq!(filter.frequency("ghost"), 0.0);
    }

    // 18. Multiple keys tracked independently
    #[test]
    fn test_multiple_keys_independent() {
        let mut filter = default_filter();
        for _ in 0..5 {
            filter.record_access("a");
        }
        filter.record_access("b");
        assert!(filter.should_admit("a"));
        assert!(!filter.should_admit("b"));
    }
}

// Helper only used in tests — avoids dead-code warnings.
#[cfg(test)]
impl AdmissionFilter {
    fn hit_rate_zero_check(&self) -> f64 {
        self.hot_tier_hit_rate()
    }
}
