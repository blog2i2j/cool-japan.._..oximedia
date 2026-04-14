//! Bandwidth-aware transcoding quality trigger.
//!
//! This module monitors live download bandwidth and signals the transcoding
//! pipeline to step its quality (bitrate/resolution) up or down so that the
//! output never exceeds what the network can reliably deliver.
//!
//! ## Design
//!
//! ```text
//!  BandwidthMonitor ──(sample)──► TriggerEvaluator ──(TriggerAction)──► Encoder
//!       ▲                                │
//!       │                                └─ TriggerHistory (audit log)
//!  bandwidth samples
//! ```
//!
//! ### Trigger algorithm
//!
//! 1. Maintain a sliding window of recent bandwidth samples (configurable depth).
//! 2. Compute a smoothed estimate (exponential moving average).
//! 3. Compare against a ladder of quality tiers (sorted by bitrate ascending).
//! 4. Emit [`TriggerAction::Downgrade`] when the EMA falls below
//!    `current_tier_bitrate * safety_factor`.
//! 5. Emit [`TriggerAction::Upgrade`] when the EMA exceeds
//!    `next_tier_bitrate * safety_factor` for at least `upgrade_hold_ms` ms.
//! 6. Emit [`TriggerAction::Hold`] otherwise (hysteresis).
//!
//! The safety factor (default 1.25) ensures the encoder never runs at a bitrate
//! so close to the available bandwidth that jitter causes rebuffering.

#![allow(dead_code)]

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::error::{NetError, NetResult};

// ─── Quality Tier ─────────────────────────────────────────────────────────────

/// A single quality tier in the encoding ladder.
///
/// Tiers must be kept sorted by `bitrate_bps` ascending so that index 0 is
/// the lowest quality.
#[derive(Debug, Clone, PartialEq)]
pub struct QualityTier {
    /// Human-readable label (e.g. `"360p"`, `"1080p60"`).
    pub label: String,
    /// Target encoded bitrate in bits per second.
    pub bitrate_bps: u64,
    /// Video width in pixels (0 if audio-only).
    pub width: u32,
    /// Video height in pixels (0 if audio-only).
    pub height: u32,
}

impl QualityTier {
    /// Creates a new quality tier.
    #[must_use]
    pub fn new(label: impl Into<String>, bitrate_bps: u64, width: u32, height: u32) -> Self {
        Self {
            label: label.into(),
            bitrate_bps,
            width,
            height,
        }
    }

    /// Creates an audio-only tier.
    #[must_use]
    pub fn audio_only(label: impl Into<String>, bitrate_bps: u64) -> Self {
        Self::new(label, bitrate_bps, 0, 0)
    }
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the bandwidth trigger.
#[derive(Debug, Clone)]
pub struct TriggerConfig {
    /// Quality tiers sorted by bitrate ascending (lowest quality first).
    pub tiers: Vec<QualityTier>,

    /// EMA smoothing factor α ∈ (0, 1].  Higher → faster adaptation.
    ///
    /// Default: 0.25.
    pub ema_alpha: f64,

    /// Safety factor applied to tier bitrates before comparison.
    ///
    /// The encoder only upgrades when `ema_bps > next_tier_bps * safety_factor`.
    /// Default: 1.25  (25 % headroom).
    pub safety_factor: f64,

    /// Minimum time the EMA must stay above the upgrade threshold before an
    /// [`TriggerAction::Upgrade`] is emitted (prevents oscillation on brief spikes).
    ///
    /// Default: 3 s.
    pub upgrade_hold: Duration,

    /// Sliding window depth for raw bandwidth samples used to compute the EMA.
    ///
    /// Default: 8.
    pub window_depth: usize,

    /// Minimum interval between consecutive downgrade events.
    ///
    /// Default: 2 s.
    pub downgrade_cooldown: Duration,
}

impl Default for TriggerConfig {
    fn default() -> Self {
        // Standard 4-rung ABR ladder
        let tiers = vec![
            QualityTier::new("240p", 400_000, 426, 240),
            QualityTier::new("480p", 1_200_000, 854, 480),
            QualityTier::new("720p", 2_500_000, 1280, 720),
            QualityTier::new("1080p", 5_000_000, 1920, 1080),
        ];
        Self {
            tiers,
            ema_alpha: 0.25,
            safety_factor: 1.25,
            upgrade_hold: Duration::from_secs(3),
            window_depth: 8,
            downgrade_cooldown: Duration::from_secs(2),
        }
    }
}

impl TriggerConfig {
    /// Validates the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`NetError::InvalidState`] if the tier list is empty, if any
    /// bitrate is zero, or if the safety factor is not in `(0, 10]`.
    pub fn validate(&self) -> NetResult<()> {
        if self.tiers.is_empty() {
            return Err(NetError::invalid_state(
                "at least one quality tier required",
            ));
        }
        for tier in &self.tiers {
            if tier.bitrate_bps == 0 {
                return Err(NetError::invalid_state(format!(
                    "tier '{}' has zero bitrate",
                    tier.label
                )));
            }
        }
        if self.safety_factor <= 0.0 || self.safety_factor > 10.0 {
            return Err(NetError::invalid_state("safety_factor must be in (0, 10]"));
        }
        if self.ema_alpha <= 0.0 || self.ema_alpha > 1.0 {
            return Err(NetError::invalid_state("ema_alpha must be in (0, 1]"));
        }
        Ok(())
    }

    /// Sorts tiers by bitrate ascending and validates.
    ///
    /// Call this after constructing a custom config.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`validate`](Self::validate).
    pub fn prepare(&mut self) -> NetResult<()> {
        self.tiers.sort_by_key(|t| t.bitrate_bps);
        self.validate()
    }
}

// ─── Trigger Action ───────────────────────────────────────────────────────────

/// The quality change action recommended by the trigger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TriggerAction {
    /// Remain at the current quality tier.
    Hold,
    /// Step down one tier (lower bitrate / resolution).
    Downgrade {
        /// Index of the target tier in [`TriggerConfig::tiers`].
        tier_index: usize,
        /// Reason string for logging.
        reason: String,
    },
    /// Step up one tier (higher bitrate / resolution).
    Upgrade {
        /// Index of the target tier in [`TriggerConfig::tiers`].
        tier_index: usize,
        /// Reason string for logging.
        reason: String,
    },
}

impl TriggerAction {
    /// Returns `true` if this action changes the current tier.
    #[must_use]
    pub const fn is_change(&self) -> bool {
        !matches!(self, Self::Hold)
    }

    /// Returns the target tier index, or `None` if [`TriggerAction::Hold`].
    #[must_use]
    pub const fn tier_index(&self) -> Option<usize> {
        match self {
            Self::Hold => None,
            Self::Downgrade { tier_index, .. } | Self::Upgrade { tier_index, .. } => {
                Some(*tier_index)
            }
        }
    }
}

// ─── Trigger History ──────────────────────────────────────────────────────────

/// An entry in the trigger audit log.
#[derive(Debug, Clone)]
pub struct TriggerEvent {
    /// Wall-clock instant when the event was emitted.
    pub when: Instant,
    /// The action that was taken.
    pub action: TriggerAction,
    /// EMA bandwidth at the time of the decision (bps).
    pub ema_bps: f64,
    /// Current tier index at the time of the decision.
    pub from_tier: usize,
}

// ─── Bandwidth Monitor ────────────────────────────────────────────────────────

/// A single raw bandwidth sample fed into the trigger.
#[derive(Debug, Clone, Copy)]
pub struct BandwidthObservation {
    /// Measured bandwidth in bits per second.
    pub bps: f64,
    /// Wall-clock instant of the measurement.
    pub measured_at: Instant,
}

impl BandwidthObservation {
    /// Creates a new observation.
    #[must_use]
    pub fn new(bps: f64) -> Self {
        Self {
            bps: bps.max(0.0),
            measured_at: Instant::now(),
        }
    }
}

// ─── BandwidthTrigger ─────────────────────────────────────────────────────────

/// Bandwidth-aware transcoding quality trigger.
///
/// Feed bandwidth observations via [`add_observation`](Self::add_observation)
/// and call [`evaluate`](Self::evaluate) after each observation to get the
/// recommended quality action.
///
/// # Example
///
/// ```rust
/// use oximedia_net::bandwidth_trigger::{BandwidthTrigger, TriggerConfig, BandwidthObservation};
///
/// let mut trigger = BandwidthTrigger::new(TriggerConfig::default()).unwrap();
/// trigger.add_observation(BandwidthObservation::new(6_000_000.0));
/// let action = trigger.evaluate();
/// // inspect action...
/// ```
#[derive(Debug)]
pub struct BandwidthTrigger {
    config: TriggerConfig,

    /// Exponential moving average of bandwidth in bps.
    ema_bps: f64,

    /// Whether the EMA has been initialised.
    ema_initialised: bool,

    /// Current quality tier index.
    current_tier: usize,

    /// Ring buffer of raw samples (used for initial EMA seeding).
    samples: VecDeque<BandwidthObservation>,

    /// When the EMA first crossed the upgrade threshold for the current level.
    upgrade_candidate_since: Option<Instant>,

    /// When the last downgrade event was emitted.
    last_downgrade: Option<Instant>,

    /// Audit history of all emitted trigger events.
    history: Vec<TriggerEvent>,
}

impl BandwidthTrigger {
    /// Creates a new trigger.
    ///
    /// Starts at the lowest quality tier and holds until enough observations
    /// accumulate.
    ///
    /// # Errors
    ///
    /// Returns [`NetError::InvalidState`] if `config.validate()` fails.
    pub fn new(mut config: TriggerConfig) -> NetResult<Self> {
        config.prepare()?;
        let window = config.window_depth;
        Ok(Self {
            config,
            ema_bps: 0.0,
            ema_initialised: false,
            current_tier: 0,
            samples: VecDeque::with_capacity(window),
            upgrade_candidate_since: None,
            last_downgrade: None,
            history: Vec::new(),
        })
    }

    /// Feeds a new bandwidth observation into the EMA.
    pub fn add_observation(&mut self, obs: BandwidthObservation) {
        if self.samples.len() == self.config.window_depth {
            self.samples.pop_front();
        }
        self.samples.push_back(obs);

        // Update EMA
        if !self.ema_initialised {
            self.ema_bps = obs.bps;
            self.ema_initialised = true;
        } else {
            let alpha = self.config.ema_alpha;
            self.ema_bps = alpha * obs.bps + (1.0 - alpha) * self.ema_bps;
        }
    }

    /// Returns the current exponential moving average bandwidth in bps.
    #[must_use]
    pub const fn ema_bps(&self) -> f64 {
        self.ema_bps
    }

    /// Returns the current quality tier index.
    #[must_use]
    pub const fn current_tier(&self) -> usize {
        self.current_tier
    }

    /// Returns a reference to the current quality tier descriptor.
    #[must_use]
    pub fn current_tier_info(&self) -> &QualityTier {
        // Safety: current_tier is always kept in bounds
        &self.config.tiers[self.current_tier]
    }

    /// Returns a slice of all quality tiers.
    #[must_use]
    pub fn tiers(&self) -> &[QualityTier] {
        &self.config.tiers
    }

    /// Returns the full event history.
    #[must_use]
    pub fn history(&self) -> &[TriggerEvent] {
        &self.history
    }

    /// Forces the trigger to a specific tier index without hysteresis.
    ///
    /// # Errors
    ///
    /// Returns [`NetError::InvalidState`] if `tier_index` is out of range.
    pub fn force_tier(&mut self, tier_index: usize) -> NetResult<()> {
        if tier_index >= self.config.tiers.len() {
            return Err(NetError::invalid_state(format!(
                "tier_index {tier_index} out of range (max {})",
                self.config.tiers.len() - 1
            )));
        }
        self.current_tier = tier_index;
        self.upgrade_candidate_since = None;
        Ok(())
    }

    /// Evaluates the current EMA and returns the recommended [`TriggerAction`].
    ///
    /// Call this after each [`add_observation`](Self::add_observation).  The
    /// result is also appended to the internal history when the action is a
    /// downgrade or upgrade.
    ///
    /// Returns [`TriggerAction::Hold`] when no observations have been added yet.
    #[must_use]
    pub fn evaluate(&mut self) -> TriggerAction {
        if !self.ema_initialised {
            return TriggerAction::Hold;
        }

        let now = Instant::now();
        let sf = self.config.safety_factor;
        let tier_count = self.config.tiers.len();
        let current_tier_bps = self.config.tiers[self.current_tier].bitrate_bps as f64;

        // ── Downgrade check ───────────────────────────────────────────────────
        // Downgrade if EMA < current tier bitrate * safety_factor
        let downgrade_threshold = current_tier_bps * sf;
        let cooldown_elapsed = self
            .last_downgrade
            .map(|t| now.duration_since(t) >= self.config.downgrade_cooldown)
            .unwrap_or(true);

        if self.ema_bps < downgrade_threshold && self.current_tier > 0 && cooldown_elapsed {
            let new_tier = self.best_tier_for(self.ema_bps);
            if new_tier < self.current_tier {
                let reason = format!(
                    "EMA {:.0} bps < threshold {:.0} bps (tier '{}' × {:.2})",
                    self.ema_bps,
                    downgrade_threshold,
                    self.config.tiers[self.current_tier].label,
                    sf,
                );
                self.last_downgrade = Some(now);
                self.upgrade_candidate_since = None;
                let action = TriggerAction::Downgrade {
                    tier_index: new_tier,
                    reason: reason.clone(),
                };
                self.record_event(now, action.clone());
                self.current_tier = new_tier;
                return action;
            }
        }

        // ── Upgrade check ─────────────────────────────────────────────────────
        if self.current_tier + 1 < tier_count {
            let next_tier_bps = self.config.tiers[self.current_tier + 1].bitrate_bps as f64;
            let upgrade_threshold = next_tier_bps * sf;

            if self.ema_bps >= upgrade_threshold {
                match self.upgrade_candidate_since {
                    None => {
                        self.upgrade_candidate_since = Some(now);
                    }
                    Some(since) => {
                        if now.duration_since(since) >= self.config.upgrade_hold {
                            let new_tier = self.current_tier + 1;
                            let reason = format!(
                                "EMA {:.0} bps ≥ threshold {:.0} bps (tier '{}' × {:.2}) for {:.1}s",
                                self.ema_bps,
                                upgrade_threshold,
                                self.config.tiers[new_tier].label,
                                sf,
                                now.duration_since(since).as_secs_f64(),
                            );
                            self.upgrade_candidate_since = None;
                            let action = TriggerAction::Upgrade {
                                tier_index: new_tier,
                                reason: reason.clone(),
                            };
                            self.record_event(now, action.clone());
                            self.current_tier = new_tier;
                            return action;
                        }
                    }
                }
            } else {
                // EMA dropped below upgrade threshold — reset hold timer
                self.upgrade_candidate_since = None;
            }
        }

        TriggerAction::Hold
    }

    /// Returns the highest tier whose bitrate fits within the available bandwidth
    /// divided by the safety factor.
    fn best_tier_for(&self, ema_bps: f64) -> usize {
        let sf = self.config.safety_factor;
        let mut best = 0;
        for (idx, tier) in self.config.tiers.iter().enumerate() {
            if tier.bitrate_bps as f64 * sf <= ema_bps {
                best = idx;
            }
        }
        best
    }

    /// Records a non-Hold event to the history.
    fn record_event(&mut self, when: Instant, action: TriggerAction) {
        self.history.push(TriggerEvent {
            when,
            action,
            ema_bps: self.ema_bps,
            from_tier: self.current_tier,
        });
    }

    /// Resets the trigger to its initial state (lowest tier, no EMA).
    pub fn reset(&mut self) {
        self.ema_bps = 0.0;
        self.ema_initialised = false;
        self.current_tier = 0;
        self.samples.clear();
        self.upgrade_candidate_since = None;
        self.last_downgrade = None;
        self.history.clear();
    }

    /// Returns a snapshot of the current trigger state for monitoring purposes.
    #[must_use]
    pub fn snapshot(&self) -> TriggerSnapshot {
        TriggerSnapshot {
            ema_bps: self.ema_bps,
            current_tier: self.current_tier,
            current_tier_label: self.config.tiers[self.current_tier].label.clone(),
            current_tier_bitrate_bps: self.config.tiers[self.current_tier].bitrate_bps,
            sample_count: self.samples.len(),
            event_count: self.history.len(),
        }
    }
}

// ─── Snapshot ─────────────────────────────────────────────────────────────────

/// A read-only snapshot of the trigger's current state.
#[derive(Debug, Clone)]
pub struct TriggerSnapshot {
    /// Current EMA bandwidth estimate in bps.
    pub ema_bps: f64,
    /// Index of the current quality tier.
    pub current_tier: usize,
    /// Label of the current quality tier.
    pub current_tier_label: String,
    /// Bitrate of the current quality tier.
    pub current_tier_bitrate_bps: u64,
    /// Number of raw samples in the window.
    pub sample_count: usize,
    /// Total number of non-Hold events recorded.
    pub event_count: usize,
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn make_trigger() -> BandwidthTrigger {
        BandwidthTrigger::new(TriggerConfig::default()).expect("valid config")
    }

    // Feed N identical observations and return the last action.
    fn feed_bps(trigger: &mut BandwidthTrigger, bps: f64, n: usize) -> TriggerAction {
        let mut action = TriggerAction::Hold;
        for _ in 0..n {
            trigger.add_observation(BandwidthObservation::new(bps));
            action = trigger.evaluate();
        }
        action
    }

    #[test]
    fn test_default_config_validates() {
        let mut cfg = TriggerConfig::default();
        cfg.prepare().expect("default config should be valid");
    }

    #[test]
    fn test_empty_tiers_rejected() {
        let mut cfg = TriggerConfig {
            tiers: vec![],
            ..TriggerConfig::default()
        };
        assert!(cfg.prepare().is_err());
    }

    #[test]
    fn test_zero_bitrate_tier_rejected() {
        let mut cfg = TriggerConfig {
            tiers: vec![QualityTier::new("bad", 0, 0, 0)],
            ..TriggerConfig::default()
        };
        assert!(cfg.prepare().is_err());
    }

    #[test]
    fn test_invalid_safety_factor_rejected() {
        let mut cfg = TriggerConfig {
            safety_factor: -1.0,
            ..TriggerConfig::default()
        };
        assert!(cfg.prepare().is_err());
    }

    #[test]
    fn test_hold_before_observations() {
        let mut trigger = make_trigger();
        assert_eq!(trigger.evaluate(), TriggerAction::Hold);
    }

    #[test]
    fn test_stays_at_lowest_tier_on_low_bandwidth() {
        let mut trigger = make_trigger();
        // 100 kbps — far below any tier
        feed_bps(&mut trigger, 100_000.0, 10);
        assert_eq!(trigger.current_tier(), 0);
    }

    #[test]
    fn test_downgrade_emitted_on_bandwidth_drop() {
        let mut trigger = make_trigger();
        // Start at 1080p tier
        trigger.force_tier(3).expect("tier 3 exists");
        // Feed very low bandwidth — should downgrade
        feed_bps(&mut trigger, 200_000.0, 10);
        assert!(trigger.current_tier() < 3, "should have downgraded");
    }

    #[test]
    fn test_upgrade_requires_hold_period() {
        let mut trigger = BandwidthTrigger::new(TriggerConfig {
            upgrade_hold: Duration::from_secs(100), // very long hold
            ..TriggerConfig::default()
        })
        .expect("valid");
        // Massive bandwidth — but hold period not met
        feed_bps(&mut trigger, 50_000_000.0, 5);
        // Should still be at tier 0 because hold period hasn't elapsed
        assert_eq!(
            trigger.current_tier(),
            0,
            "upgrade should not have fired yet"
        );
    }

    #[test]
    fn test_upgrade_fires_after_hold() {
        let mut trigger = BandwidthTrigger::new(TriggerConfig {
            upgrade_hold: Duration::ZERO, // instant upgrade
            ..TriggerConfig::default()
        })
        .expect("valid");
        // Bandwidth well above 480p * 1.25 = 1.5 Mbps
        feed_bps(&mut trigger, 2_000_000.0, 5);
        // With zero hold, should upgrade at least once
        assert!(trigger.current_tier() > 0, "should have upgraded");
    }

    #[test]
    fn test_ema_initialised_from_first_sample() {
        let mut trigger = make_trigger();
        trigger.add_observation(BandwidthObservation::new(3_000_000.0));
        assert!((trigger.ema_bps() - 3_000_000.0).abs() < 1.0);
    }

    #[test]
    fn test_ema_smoothing() {
        let mut trigger = make_trigger();
        trigger.add_observation(BandwidthObservation::new(10_000_000.0));
        trigger.add_observation(BandwidthObservation::new(0.0)); // sudden drop
                                                                 // EMA should be between 0 and 10 Mbps
        assert!(trigger.ema_bps() > 0.0);
        assert!(trigger.ema_bps() < 10_000_000.0);
    }

    #[test]
    fn test_force_tier_out_of_range_errors() {
        let mut trigger = make_trigger();
        assert!(trigger.force_tier(99).is_err());
    }

    #[test]
    fn test_snapshot_reflects_state() {
        let mut trigger = make_trigger();
        trigger.add_observation(BandwidthObservation::new(5_000_000.0));
        let snap = trigger.snapshot();
        assert_eq!(snap.sample_count, 1);
        assert!(snap.ema_bps > 0.0);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut trigger = make_trigger();
        feed_bps(&mut trigger, 5_000_000.0, 5);
        trigger.reset();
        assert_eq!(trigger.snapshot().sample_count, 0);
        assert_eq!(trigger.snapshot().ema_bps, 0.0);
        assert_eq!(trigger.current_tier(), 0);
    }

    #[test]
    fn test_history_records_events() {
        let mut trigger = BandwidthTrigger::new(TriggerConfig {
            upgrade_hold: Duration::ZERO,
            ..TriggerConfig::default()
        })
        .expect("valid");
        // Force high bandwidth → should get at least one upgrade event
        feed_bps(&mut trigger, 10_000_000.0, 20);
        assert!(
            !trigger.history().is_empty(),
            "expected at least one event in history"
        );
    }

    #[test]
    fn test_tier_label() {
        let trigger = make_trigger();
        let info = trigger.current_tier_info();
        assert!(!info.label.is_empty());
    }

    #[test]
    fn test_quality_tier_audio_only() {
        let tier = QualityTier::audio_only("AAC 128k", 128_000);
        assert_eq!(tier.width, 0);
        assert_eq!(tier.height, 0);
        assert_eq!(tier.bitrate_bps, 128_000);
    }

    #[test]
    fn test_trigger_action_is_change() {
        assert!(!TriggerAction::Hold.is_change());
        assert!(TriggerAction::Downgrade {
            tier_index: 0,
            reason: String::new()
        }
        .is_change());
        assert!(TriggerAction::Upgrade {
            tier_index: 1,
            reason: String::new()
        }
        .is_change());
    }

    #[test]
    fn test_trigger_action_tier_index() {
        assert_eq!(TriggerAction::Hold.tier_index(), None);
        assert_eq!(
            TriggerAction::Upgrade {
                tier_index: 2,
                reason: String::new()
            }
            .tier_index(),
            Some(2)
        );
    }
}
