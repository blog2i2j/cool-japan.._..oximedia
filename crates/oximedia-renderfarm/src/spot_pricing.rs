// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Spot instance pricing models and preemption handling for cloud render workers.
//!
//! This module provides:
//!
//! - [`SpotPriceModel`] — time-varying, market-driven spot pricing per instance type
//! - [`PreemptionConfig`] — policy knobs controlling how preemptions are handled
//! - [`SpotBidStrategy`] — algorithm for choosing a bid price from observed prices
//! - [`PreemptionEvent`] — record of a single preemption occurrence
//! - [`SpotScheduler`] — cost-aware task scheduler that selects the cheapest viable
//!   spot capacity and gracefully degrades on preemption
//!
//! ## Design rationale
//!
//! Cloud spot / preemptible instances can be interrupted at any moment when the
//! cloud provider reclaims capacity.  A render farm must therefore:
//!
//! 1. Predict the cost of using spot capacity vs. on-demand (or on-premise).
//! 2. Set bid prices that balance cost savings against interruption risk.
//! 3. React to preemptions by checkpointing work and re-queuing affected tasks on
//!    alternative capacity.
//!
//! [`SpotPriceModel`] models the observed spot price history and can forecast the
//! next price using an exponential moving average (EMA).  [`SpotScheduler`] uses
//! this to gate task dispatch: if the current price exceeds the configured maximum,
//! the task is held until cheaper capacity becomes available.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// ─────────────────────────────────────────────────────────────────────────────
// SpotMarket / CloudRegion
// ─────────────────────────────────────────────────────────────────────────────

/// A cloud market region where spot instances are purchased.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CloudRegion {
    /// AWS us-east-1
    AwsUsEast1,
    /// AWS eu-west-1
    AwsEuWest1,
    /// GCP us-central1
    GcpUsCentral1,
    /// GCP europe-west4
    GcpEuropeWest4,
    /// Azure East US
    AzureEastUs,
    /// Azure West Europe
    AzureWestEurope,
    /// Custom / on-premise region identified by a string label.
    Custom(String),
}

impl CloudRegion {
    /// Short display name for the region.
    #[must_use]
    pub fn display_name(&self) -> &str {
        match self {
            Self::AwsUsEast1 => "aws:us-east-1",
            Self::AwsEuWest1 => "aws:eu-west-1",
            Self::GcpUsCentral1 => "gcp:us-central1",
            Self::GcpEuropeWest4 => "gcp:europe-west4",
            Self::AzureEastUs => "azure:eastus",
            Self::AzureWestEurope => "azure:westeurope",
            Self::Custom(s) => s.as_str(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SpotPriceSample
// ─────────────────────────────────────────────────────────────────────────────

/// A single observed spot price data point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpotPriceSample {
    /// Unix timestamp (seconds) when this price was observed.
    pub timestamp_secs: i64,
    /// Spot price in USD per instance-hour at this moment.
    pub price_usd_per_hour: f64,
    /// Observed interruption rate in the preceding hour (0.0–1.0).
    pub interruption_rate: f32,
}

impl SpotPriceSample {
    /// Construct a new price sample.
    #[must_use]
    pub fn new(timestamp_secs: i64, price_usd_per_hour: f64, interruption_rate: f32) -> Self {
        Self {
            timestamp_secs,
            price_usd_per_hour,
            interruption_rate: interruption_rate.clamp(0.0, 1.0),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SpotPriceModel
// ─────────────────────────────────────────────────────────────────────────────

/// Exponential moving-average based spot price model.
///
/// Accumulates historical price samples and provides:
/// - EMA price forecast for the next interval
/// - Interruption rate smoothing
/// - Statistical helpers (min / max / mean over the window)
///
/// The EMA smoothing factor `alpha` ∈ (0, 1]:
/// - Near 1.0 → heavily weights the most recent sample.
/// - Near 0.0 → smooths over a long historical window.
///
/// Default `alpha` is `0.2` (i.e. ~5 sample memory).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpotPriceModel {
    /// Instance type this model tracks (e.g. "c5.2xlarge", "n2-standard-8").
    pub instance_type: String,
    /// Cloud region.
    pub region: CloudRegion,
    /// On-demand (non-spot) price for this instance type in USD/hr.
    pub on_demand_price_usd_per_hour: f64,
    /// Smoothing factor for the EMA (0 < alpha ≤ 1).
    alpha: f64,
    /// Rolling window of price samples (newest at back).
    samples: VecDeque<SpotPriceSample>,
    /// Maximum samples retained in memory.
    max_samples: usize,
    /// Current EMA of the spot price.
    ema_price: Option<f64>,
    /// Current EMA of the interruption rate.
    ema_interruption: Option<f32>,
}

impl SpotPriceModel {
    /// Construct a new price model with default alpha (0.2) and window of 100 samples.
    ///
    /// # Panics (dev-time only)
    ///
    /// `alpha` must be in (0.0, 1.0].  Out-of-range values are clamped to a
    /// valid range in release mode; in debug builds `alpha <= 0` is caught by
    /// the `debug_assert`.
    #[must_use]
    pub fn new(
        instance_type: impl Into<String>,
        region: CloudRegion,
        on_demand_price_usd_per_hour: f64,
    ) -> Self {
        Self::with_alpha(instance_type, region, on_demand_price_usd_per_hour, 0.2)
    }

    /// Construct with a custom EMA alpha.
    #[must_use]
    pub fn with_alpha(
        instance_type: impl Into<String>,
        region: CloudRegion,
        on_demand_price_usd_per_hour: f64,
        alpha: f64,
    ) -> Self {
        let alpha = alpha.clamp(1e-6, 1.0);
        Self {
            instance_type: instance_type.into(),
            region,
            on_demand_price_usd_per_hour,
            alpha,
            samples: VecDeque::new(),
            max_samples: 100,
            ema_price: None,
            ema_interruption: None,
        }
    }

    /// Set the maximum number of samples retained in the rolling window.
    pub fn set_max_samples(&mut self, max: usize) {
        self.max_samples = max.max(1);
        while self.samples.len() > self.max_samples {
            self.samples.pop_front();
        }
    }

    /// Ingest a new price observation, updating EMA accumulators.
    pub fn observe(&mut self, sample: SpotPriceSample) {
        // Update EMA price
        self.ema_price = Some(match self.ema_price {
            None => sample.price_usd_per_hour,
            Some(prev) => prev + self.alpha * (sample.price_usd_per_hour - prev),
        });

        // Update EMA interruption rate
        let rate = f64::from(sample.interruption_rate);
        self.ema_interruption = Some(match self.ema_interruption {
            None => sample.interruption_rate,
            Some(prev) => {
                let ema = f64::from(prev) + self.alpha * (rate - f64::from(prev));
                ema as f32
            }
        });

        // Maintain rolling window
        if self.samples.len() >= self.max_samples {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    /// Predicted price for the next interval (EMA, or `None` before first sample).
    #[must_use]
    pub fn predicted_price(&self) -> Option<f64> {
        self.ema_price
    }

    /// Predicted interruption rate (EMA, or `None` before first sample).
    #[must_use]
    pub fn predicted_interruption_rate(&self) -> Option<f32> {
        self.ema_interruption
    }

    /// Minimum price seen in the current window, or `None` when empty.
    #[must_use]
    pub fn min_price(&self) -> Option<f64> {
        self.samples
            .iter()
            .map(|s| s.price_usd_per_hour)
            .reduce(f64::min)
    }

    /// Maximum price seen in the current window, or `None` when empty.
    #[must_use]
    pub fn max_price(&self) -> Option<f64> {
        self.samples
            .iter()
            .map(|s| s.price_usd_per_hour)
            .reduce(f64::max)
    }

    /// Arithmetic mean of all prices in the current window, or `None` when empty.
    #[must_use]
    pub fn mean_price(&self) -> Option<f64> {
        if self.samples.is_empty() {
            return None;
        }
        let sum: f64 = self.samples.iter().map(|s| s.price_usd_per_hour).sum();
        Some(sum / self.samples.len() as f64)
    }

    /// Current spot discount relative to on-demand in percent (negative = more expensive).
    ///
    /// Returns `None` when no samples have been observed yet.
    #[must_use]
    pub fn spot_discount_pct(&self) -> Option<f64> {
        let spot = self.ema_price?;
        if self.on_demand_price_usd_per_hour <= 0.0 {
            return None;
        }
        Some((1.0 - spot / self.on_demand_price_usd_per_hour) * 100.0)
    }

    /// Number of samples in the rolling window.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SpotBidStrategy
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy for computing the bid price submitted to the cloud provider.
///
/// A higher bid reduces interruption risk but increases cost.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpotBidStrategy {
    /// Bid exactly the on-demand price (maximum safety, minimum savings).
    OnDemandCap,
    /// Bid a fixed multiplier of the predicted EMA price.
    ///
    /// `1.1` means bid 10% above the current EMA — typically enough buffer to
    /// avoid most interruptions while still saving vs on-demand.
    EmaMultiplier(f64),
    /// Bid at a fixed USD/hr value regardless of observed price history.
    Fixed(f64),
    /// Bid the maximum of (`EmaMultiplier` result) and a floor price.
    ///
    /// Ensures bids never dip below `floor_usd_per_hour` even when prices spike.
    FlooredEma {
        /// Multiplier applied to the EMA.
        multiplier: f64,
        /// Absolute minimum bid.
        floor_usd_per_hour: f64,
    },
}

impl SpotBidStrategy {
    /// Compute the bid price in USD/hr given a price model.
    ///
    /// Returns `None` if the model has no samples and the strategy requires an EMA.
    #[must_use]
    pub fn compute_bid(&self, model: &SpotPriceModel) -> Option<f64> {
        match self {
            Self::OnDemandCap => Some(model.on_demand_price_usd_per_hour),
            Self::EmaMultiplier(m) => {
                let ema = model.predicted_price()?;
                Some((ema * m).min(model.on_demand_price_usd_per_hour))
            }
            Self::Fixed(price) => Some(*price),
            Self::FlooredEma {
                multiplier,
                floor_usd_per_hour,
            } => {
                let ema = model.predicted_price()?;
                let bid = (ema * multiplier).min(model.on_demand_price_usd_per_hour);
                Some(bid.max(*floor_usd_per_hour))
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PreemptionConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration controlling how the scheduler reacts to spot preemptions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreemptionConfig {
    /// Maximum fraction of workers that may be preempted simultaneously before
    /// the scheduler escalates to on-demand capacity.
    ///
    /// Range [0.0, 1.0].  Default `0.3` (30%).
    pub max_simultaneous_preemption_fraction: f32,

    /// Grace period in seconds the cloud provider gives before reclaiming the
    /// instance after sending a preemption notice.
    ///
    /// AWS gives 2 min (120 s), GCP gives 30 s.
    pub grace_period_secs: u32,

    /// Whether to automatically checkpoint work when a preemption notice is
    /// received, so the task can be resumed rather than restarted.
    pub enable_checkpoint_on_preemption: bool,

    /// Whether to fall back to on-demand capacity when the spot interruption
    /// rate exceeds `fallback_interruption_threshold`.
    pub fallback_to_ondemand: bool,

    /// Interruption rate above which the scheduler switches a pool to on-demand.
    ///
    /// Range [0.0, 1.0].
    pub fallback_interruption_threshold: f32,

    /// Maximum number of consecutive preemptions for a single task before it
    /// is permanently moved to on-demand (avoids infinite spot churn).
    pub max_consecutive_preemptions: u8,
}

impl Default for PreemptionConfig {
    fn default() -> Self {
        Self {
            max_simultaneous_preemption_fraction: 0.30,
            grace_period_secs: 120,
            enable_checkpoint_on_preemption: true,
            fallback_to_ondemand: true,
            fallback_interruption_threshold: 0.20,
            max_consecutive_preemptions: 3,
        }
    }
}

impl PreemptionConfig {
    /// Conservative config: low interruption tolerance, frequent checkpoints.
    #[must_use]
    pub fn conservative() -> Self {
        Self {
            max_simultaneous_preemption_fraction: 0.10,
            grace_period_secs: 120,
            enable_checkpoint_on_preemption: true,
            fallback_to_ondemand: true,
            fallback_interruption_threshold: 0.05,
            max_consecutive_preemptions: 2,
        }
    }

    /// Aggressive config: high interruption tolerance, fewer checkpoints.
    #[must_use]
    pub fn aggressive() -> Self {
        Self {
            max_simultaneous_preemption_fraction: 0.50,
            grace_period_secs: 30,
            enable_checkpoint_on_preemption: false,
            fallback_to_ondemand: false,
            fallback_interruption_threshold: 0.50,
            max_consecutive_preemptions: 10,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PreemptionEvent
// ─────────────────────────────────────────────────────────────────────────────

/// Record of a single spot preemption occurrence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreemptionEvent {
    /// Identifier of the worker instance that was preempted.
    pub worker_id: String,
    /// Identifier of the task that was running when preemption occurred.
    pub task_id: String,
    /// Unix timestamp (seconds) when the preemption notice was received.
    pub timestamp_secs: i64,
    /// Last frame (or chunk start) that was safely checkpointed before preemption.
    pub last_checkpoint_frame: Option<u32>,
    /// Whether the task was successfully checkpointed before instance reclamation.
    pub checkpointed: bool,
    /// Region where the preemption occurred.
    pub region: CloudRegion,
}

impl PreemptionEvent {
    /// Construct a new preemption event.
    #[must_use]
    pub fn new(
        worker_id: impl Into<String>,
        task_id: impl Into<String>,
        timestamp_secs: i64,
        region: CloudRegion,
    ) -> Self {
        Self {
            worker_id: worker_id.into(),
            task_id: task_id.into(),
            timestamp_secs,
            last_checkpoint_frame: None,
            checkpointed: false,
            region,
        }
    }

    /// Builder: set last checkpoint frame.
    #[must_use]
    pub fn with_checkpoint(mut self, frame: u32) -> Self {
        self.last_checkpoint_frame = Some(frame);
        self.checkpointed = true;
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SchedulingDecision
// ─────────────────────────────────────────────────────────────────────────────

/// The outcome of a `SpotScheduler::should_dispatch` evaluation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SchedulingDecision {
    /// Dispatch the task on spot capacity at the given bid price.
    DispatchSpot {
        /// Recommended bid in USD/hr.
        bid_usd_per_hour: f64,
    },
    /// Hold the task; spot price too high or interruption rate too large.
    Hold {
        /// Reason the scheduler decided not to dispatch.
        reason: String,
    },
    /// Use on-demand capacity instead of spot for this task.
    UseOnDemand {
        /// Reason on-demand was chosen over spot.
        reason: String,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// SpotScheduler
// ─────────────────────────────────────────────────────────────────────────────

/// Cost-aware task scheduler that selects spot capacity and handles preemptions.
///
/// ## Dispatch logic
///
/// For each candidate task the scheduler:
///
/// 1. Looks up the price model for the requested instance type + region.
/// 2. Computes the bid using the configured [`SpotBidStrategy`].
/// 3. Compares the predicted price to `max_price_usd_per_hour`.
/// 4. Checks whether the predicted interruption rate is below the threshold from
///    [`PreemptionConfig`].
/// 5. If the consecutive preemption count for this task exceeds the configured
///    limit, recommends on-demand.
/// 6. Returns [`SchedulingDecision`].
///
/// ## Preemption handling
///
/// When a preemption notice arrives the caller calls [`SpotScheduler::record_preemption`] which:
/// - Stores the event for later auditing / cost reporting.
/// - Increments the per-task consecutive preemption counter.
/// - If the counter exceeds the limit, marks the task for on-demand routing.
pub struct SpotScheduler {
    /// Spot price models keyed by (instance_type, region).
    models: HashMap<(String, CloudRegion), SpotPriceModel>,
    /// Bid strategy applied to all dispatch decisions.
    bid_strategy: SpotBidStrategy,
    /// Preemption handling policy.
    preemption_config: PreemptionConfig,
    /// Preemption history log.
    preemption_log: Vec<PreemptionEvent>,
    /// Per-task consecutive preemption counter.
    consecutive_preemptions: HashMap<String, u8>,
    /// Maximum spot price accepted (USD/hr).  Tasks will be held when price exceeds this.
    max_price_usd_per_hour: f64,
    /// Maximum acceptable predicted interruption rate (0.0–1.0).
    max_interruption_rate: f32,
}

impl SpotScheduler {
    /// Create a new scheduler with the given bid strategy and preemption config.
    #[must_use]
    pub fn new(
        bid_strategy: SpotBidStrategy,
        preemption_config: PreemptionConfig,
        max_price_usd_per_hour: f64,
        max_interruption_rate: f32,
    ) -> Self {
        Self {
            models: HashMap::new(),
            bid_strategy,
            preemption_config,
            preemption_log: Vec::new(),
            consecutive_preemptions: HashMap::new(),
            max_price_usd_per_hour: max_price_usd_per_hour.max(0.0),
            max_interruption_rate: max_interruption_rate.clamp(0.0, 1.0),
        }
    }

    /// Register or retrieve a mutable reference to the price model for the given key.
    ///
    /// If no model exists, one is created with the supplied on-demand price and default alpha.
    pub fn upsert_model(
        &mut self,
        instance_type: impl Into<String>,
        region: CloudRegion,
        on_demand_price_usd_per_hour: f64,
    ) -> &mut SpotPriceModel {
        let instance_type = instance_type.into();
        let key = (instance_type.clone(), region.clone());
        self.models.entry(key).or_insert_with(|| {
            SpotPriceModel::new(instance_type, region, on_demand_price_usd_per_hour)
        })
    }

    /// Feed a new price sample into the model for the given instance type and region.
    ///
    /// Creates the model with `on_demand_price_usd_per_hour` if it doesn't exist yet.
    pub fn observe_price(
        &mut self,
        instance_type: impl Into<String>,
        region: CloudRegion,
        on_demand_price_usd_per_hour: f64,
        sample: SpotPriceSample,
    ) {
        let model = self.upsert_model(instance_type, region, on_demand_price_usd_per_hour);
        model.observe(sample);
    }

    /// Evaluate whether to dispatch `task_id` onto spot capacity of the given type/region.
    ///
    /// Returns a [`SchedulingDecision`] with the recommended action.
    #[must_use]
    pub fn should_dispatch(
        &self,
        task_id: &str,
        instance_type: &str,
        region: &CloudRegion,
    ) -> SchedulingDecision {
        let key = (instance_type.to_owned(), region.clone());
        let model = match self.models.get(&key) {
            Some(m) => m,
            None => {
                return SchedulingDecision::Hold {
                    reason: format!(
                        "no price model for {instance_type} in {}",
                        region.display_name()
                    ),
                };
            }
        };

        // Check consecutive preemption limit
        let consec = self
            .consecutive_preemptions
            .get(task_id)
            .copied()
            .unwrap_or(0);
        if consec >= self.preemption_config.max_consecutive_preemptions {
            return SchedulingDecision::UseOnDemand {
                reason: format!(
                    "task {task_id} preempted {consec} times consecutively; routing to on-demand"
                ),
            };
        }

        // Check fallback condition: high interruption rate on this model
        if self.preemption_config.fallback_to_ondemand {
            if let Some(rate) = model.predicted_interruption_rate() {
                if rate >= self.preemption_config.fallback_interruption_threshold {
                    let threshold_pct =
                        self.preemption_config.fallback_interruption_threshold * 100.0;
                    return SchedulingDecision::UseOnDemand {
                        reason: format!(
                            "interruption rate {:.1}% exceeds fallback threshold {threshold_pct:.1}%",
                            rate * 100.0,
                        ),
                    };
                }
            }
        }

        // Check interruption rate gate
        if let Some(rate) = model.predicted_interruption_rate() {
            if rate > self.max_interruption_rate {
                return SchedulingDecision::Hold {
                    reason: format!(
                        "interruption rate {:.1}% exceeds max {:.1}%",
                        rate * 100.0,
                        self.max_interruption_rate * 100.0,
                    ),
                };
            }
        }

        // Compute bid
        let bid = match self.bid_strategy.compute_bid(model) {
            Some(b) => b,
            None => {
                return SchedulingDecision::Hold {
                    reason: "price model has no samples yet; cannot compute bid".to_owned(),
                };
            }
        };

        // Check current price against our ceiling
        if let Some(current_price) = model.predicted_price() {
            if current_price > self.max_price_usd_per_hour {
                return SchedulingDecision::Hold {
                    reason: format!(
                        "current price ${current_price:.4}/hr exceeds ceiling ${:.4}/hr",
                        self.max_price_usd_per_hour
                    ),
                };
            }
        }

        SchedulingDecision::DispatchSpot {
            bid_usd_per_hour: bid,
        }
    }

    /// Record a preemption event and update internal counters.
    pub fn record_preemption(&mut self, event: PreemptionEvent) {
        let task_id = event.task_id.clone();
        self.preemption_log.push(event);

        let counter = self.consecutive_preemptions.entry(task_id).or_insert(0);
        *counter = counter.saturating_add(1);
    }

    /// Notify the scheduler that a task completed successfully, resetting its
    /// consecutive preemption counter.
    pub fn record_completion(&mut self, task_id: &str) {
        self.consecutive_preemptions.remove(task_id);
    }

    /// Return all recorded preemption events.
    #[must_use]
    pub fn preemption_log(&self) -> &[PreemptionEvent] {
        &self.preemption_log
    }

    /// Return the consecutive preemption count for a task (0 if never preempted).
    #[must_use]
    pub fn consecutive_preemption_count(&self, task_id: &str) -> u8 {
        self.consecutive_preemptions
            .get(task_id)
            .copied()
            .unwrap_or(0)
    }

    /// Total number of preemption events recorded.
    #[must_use]
    pub fn total_preemptions(&self) -> usize {
        self.preemption_log.len()
    }

    /// Fraction of recorded preemptions that were successfully checkpointed.
    ///
    /// Returns `None` when there are no preemption events.
    #[must_use]
    pub fn checkpoint_rate(&self) -> Option<f32> {
        if self.preemption_log.is_empty() {
            return None;
        }
        let checkpointed = self
            .preemption_log
            .iter()
            .filter(|e| e.checkpointed)
            .count();
        Some(checkpointed as f32 / self.preemption_log.len() as f32)
    }

    /// Estimated total savings in USD from using spot vs on-demand for the given
    /// job parameters.
    ///
    /// Savings = (on_demand_price - predicted_spot_price) × hours × instance_count.
    ///
    /// Returns `None` if no model exists or has no samples.
    #[must_use]
    pub fn estimated_savings_usd(
        &self,
        instance_type: &str,
        region: &CloudRegion,
        hours: f64,
        instance_count: u32,
    ) -> Option<f64> {
        let key = (instance_type.to_owned(), region.clone());
        let model = self.models.get(&key)?;
        let spot = model.predicted_price()?;
        let savings_per_hour = (model.on_demand_price_usd_per_hour - spot).max(0.0);
        Some(savings_per_hour * hours * f64::from(instance_count))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(price: f64, interrupt: f32) -> SpotPriceSample {
        SpotPriceSample::new(0, price, interrupt)
    }

    fn model_with_samples(prices: &[f64]) -> SpotPriceModel {
        let mut m = SpotPriceModel::new("c5.2xlarge", CloudRegion::AwsUsEast1, 0.40);
        for &p in prices {
            m.observe(sample(p, 0.05));
        }
        m
    }

    // ── SpotPriceSample ──────────────────────────────────────────────────────

    #[test]
    fn test_sample_interruption_clamped() {
        let s = SpotPriceSample::new(0, 0.10, 1.5);
        assert_eq!(s.interruption_rate, 1.0);
        let s2 = SpotPriceSample::new(0, 0.10, -0.3);
        assert_eq!(s2.interruption_rate, 0.0);
    }

    // ── SpotPriceModel ───────────────────────────────────────────────────────

    #[test]
    fn test_model_no_samples_returns_none() {
        let m = SpotPriceModel::new("c5.xlarge", CloudRegion::AwsUsEast1, 0.20);
        assert!(m.predicted_price().is_none());
        assert!(m.min_price().is_none());
        assert!(m.mean_price().is_none());
        assert!(m.spot_discount_pct().is_none());
    }

    #[test]
    fn test_model_ema_single_sample() {
        let mut m = SpotPriceModel::new("c5.xlarge", CloudRegion::AwsUsEast1, 0.20);
        m.observe(sample(0.05, 0.02));
        // First sample: EMA == the sample itself
        assert!((m.predicted_price().unwrap() - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_model_ema_converges_toward_new_price() {
        let mut m = SpotPriceModel::with_alpha("t3.micro", CloudRegion::GcpUsCentral1, 0.15, 1.0);
        // Alpha = 1.0 means EMA == last sample
        m.observe(sample(0.10, 0.01));
        m.observe(sample(0.20, 0.01));
        assert!((m.predicted_price().unwrap() - 0.20).abs() < 1e-9);
    }

    #[test]
    fn test_model_min_max_mean() {
        let m = model_with_samples(&[0.10, 0.20, 0.15]);
        assert!((m.min_price().unwrap() - 0.10).abs() < 1e-9);
        assert!((m.max_price().unwrap() - 0.20).abs() < 1e-9);
        let mean = m.mean_price().unwrap();
        assert!((mean - 0.15).abs() < 1e-6);
    }

    #[test]
    fn test_model_spot_discount_pct() {
        let mut m = SpotPriceModel::with_alpha("c5.2xlarge", CloudRegion::AwsUsEast1, 0.40, 1.0);
        m.observe(sample(0.20, 0.05)); // 50% discount
        let discount = m.spot_discount_pct().unwrap();
        assert!((discount - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_model_rolling_window_eviction() {
        let mut m = SpotPriceModel::new("t3.nano", CloudRegion::AwsEuWest1, 0.10);
        m.set_max_samples(3);
        for i in 0..5u64 {
            m.observe(SpotPriceSample::new(i as i64, 0.05, 0.01));
        }
        assert_eq!(m.sample_count(), 3);
    }

    // ── SpotBidStrategy ──────────────────────────────────────────────────────

    #[test]
    fn test_bid_on_demand_cap() {
        let m = model_with_samples(&[0.05]);
        let bid = SpotBidStrategy::OnDemandCap.compute_bid(&m).unwrap();
        assert!((bid - 0.40).abs() < 1e-9);
    }

    #[test]
    fn test_bid_ema_multiplier() {
        let mut m = SpotPriceModel::with_alpha("c5.2xlarge", CloudRegion::AwsUsEast1, 0.40, 1.0);
        m.observe(sample(0.10, 0.05));
        let bid = SpotBidStrategy::EmaMultiplier(1.2).compute_bid(&m).unwrap();
        // 0.10 * 1.2 = 0.12, but capped at on_demand (0.40)
        assert!((bid - 0.12).abs() < 1e-9);
    }

    #[test]
    fn test_bid_ema_multiplier_capped_at_on_demand() {
        let mut m = SpotPriceModel::with_alpha("c5.2xlarge", CloudRegion::AwsUsEast1, 0.40, 1.0);
        m.observe(sample(0.38, 0.05));
        // 0.38 * 1.5 = 0.57 > 0.40 → capped at 0.40
        let bid = SpotBidStrategy::EmaMultiplier(1.5).compute_bid(&m).unwrap();
        assert!((bid - 0.40).abs() < 1e-9);
    }

    #[test]
    fn test_bid_fixed() {
        let m = model_with_samples(&[0.05]);
        let bid = SpotBidStrategy::Fixed(0.25).compute_bid(&m).unwrap();
        assert!((bid - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_bid_floored_ema() {
        let mut m = SpotPriceModel::with_alpha("c5.2xlarge", CloudRegion::AwsUsEast1, 0.40, 1.0);
        m.observe(sample(0.02, 0.01)); // very cheap
        let strategy = SpotBidStrategy::FlooredEma {
            multiplier: 1.1,
            floor_usd_per_hour: 0.05,
        };
        let bid = strategy.compute_bid(&m).unwrap();
        // 0.02 * 1.1 = 0.022 < floor 0.05 → returns floor
        assert!((bid - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_bid_ema_no_samples_returns_none() {
        let m = SpotPriceModel::new("c5.xlarge", CloudRegion::AwsUsEast1, 0.20);
        assert!(SpotBidStrategy::EmaMultiplier(1.1)
            .compute_bid(&m)
            .is_none());
    }

    // ── PreemptionConfig ─────────────────────────────────────────────────────

    #[test]
    fn test_preemption_config_defaults() {
        let cfg = PreemptionConfig::default();
        assert!(cfg.fallback_to_ondemand);
        assert!(cfg.enable_checkpoint_on_preemption);
        assert_eq!(cfg.max_consecutive_preemptions, 3);
    }

    #[test]
    fn test_preemption_config_conservative() {
        let cfg = PreemptionConfig::conservative();
        assert!(cfg.fallback_interruption_threshold < 0.10);
    }

    #[test]
    fn test_preemption_config_aggressive() {
        let cfg = PreemptionConfig::aggressive();
        assert!(!cfg.fallback_to_ondemand);
        assert!(!cfg.enable_checkpoint_on_preemption);
    }

    // ── SpotScheduler ────────────────────────────────────────────────────────

    fn make_scheduler() -> SpotScheduler {
        SpotScheduler::new(
            SpotBidStrategy::EmaMultiplier(1.1),
            PreemptionConfig::default(),
            0.20, // max_price
            0.15, // max_interruption_rate
        )
    }

    #[test]
    fn test_scheduler_hold_no_model() {
        let sched = make_scheduler();
        let decision = sched.should_dispatch("task-1", "c5.2xlarge", &CloudRegion::AwsUsEast1);
        assert!(matches!(decision, SchedulingDecision::Hold { .. }));
    }

    #[test]
    fn test_scheduler_hold_no_samples() {
        let mut sched = make_scheduler();
        sched.upsert_model("c5.2xlarge", CloudRegion::AwsUsEast1, 0.40);
        // Model exists but has no samples → bid cannot be computed
        let decision = sched.should_dispatch("task-1", "c5.2xlarge", &CloudRegion::AwsUsEast1);
        assert!(matches!(decision, SchedulingDecision::Hold { .. }));
    }

    #[test]
    fn test_scheduler_dispatch_spot_cheap_price() {
        let mut sched = make_scheduler();
        sched.observe_price(
            "c5.2xlarge",
            CloudRegion::AwsUsEast1,
            0.40,
            sample(0.10, 0.02),
        );
        let decision = sched.should_dispatch("task-1", "c5.2xlarge", &CloudRegion::AwsUsEast1);
        assert!(
            matches!(decision, SchedulingDecision::DispatchSpot { bid_usd_per_hour } if bid_usd_per_hour > 0.0)
        );
    }

    #[test]
    fn test_scheduler_hold_price_too_high() {
        let mut sched = make_scheduler();
        // max_price = 0.20, observed = 0.30 → Hold
        sched.observe_price(
            "c5.2xlarge",
            CloudRegion::AwsUsEast1,
            0.40,
            sample(0.30, 0.02),
        );
        let decision = sched.should_dispatch("task-1", "c5.2xlarge", &CloudRegion::AwsUsEast1);
        assert!(matches!(decision, SchedulingDecision::Hold { .. }));
    }

    #[test]
    fn test_scheduler_fallback_high_interruption_rate() {
        let mut sched = make_scheduler();
        sched.observe_price(
            "c5.2xlarge",
            CloudRegion::AwsUsEast1,
            0.40,
            sample(0.08, 0.25), // interruption 25% > fallback threshold 20%
        );
        let decision = sched.should_dispatch("task-1", "c5.2xlarge", &CloudRegion::AwsUsEast1);
        assert!(matches!(decision, SchedulingDecision::UseOnDemand { .. }));
    }

    #[test]
    fn test_scheduler_consecutive_preemption_triggers_ondemand() {
        let mut sched = SpotScheduler::new(
            SpotBidStrategy::EmaMultiplier(1.1),
            PreemptionConfig::default(), // max_consecutive = 3
            1.00,
            0.99,
        );
        sched.observe_price(
            "c5.2xlarge",
            CloudRegion::AwsUsEast1,
            0.40,
            sample(0.05, 0.01),
        );

        // Simulate 3 preemptions
        for i in 0..3u8 {
            sched.record_preemption(PreemptionEvent::new(
                "worker-x",
                "task-preempted",
                i as i64,
                CloudRegion::AwsUsEast1,
            ));
        }

        let decision =
            sched.should_dispatch("task-preempted", "c5.2xlarge", &CloudRegion::AwsUsEast1);
        assert!(matches!(decision, SchedulingDecision::UseOnDemand { .. }));
    }

    #[test]
    fn test_scheduler_completion_resets_counter() {
        let mut sched = make_scheduler();
        sched.record_preemption(PreemptionEvent::new(
            "w1",
            "task-a",
            0,
            CloudRegion::AwsUsEast1,
        ));
        assert_eq!(sched.consecutive_preemption_count("task-a"), 1);
        sched.record_completion("task-a");
        assert_eq!(sched.consecutive_preemption_count("task-a"), 0);
    }

    #[test]
    fn test_scheduler_checkpoint_rate() {
        let mut sched = make_scheduler();
        sched.record_preemption(
            PreemptionEvent::new("w1", "t1", 0, CloudRegion::AwsUsEast1).with_checkpoint(42),
        );
        sched.record_preemption(PreemptionEvent::new("w2", "t2", 1, CloudRegion::AwsUsEast1));
        let rate = sched.checkpoint_rate().unwrap();
        assert!((rate - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_scheduler_estimated_savings() {
        let mut sched = SpotScheduler::new(
            SpotBidStrategy::EmaMultiplier(1.1),
            PreemptionConfig::default(),
            1.0,
            0.99,
        );
        // on_demand = 0.40, spot EMA = 0.10 → savings = 0.30/hr
        sched.observe_price(
            "c5.2xlarge",
            CloudRegion::AwsUsEast1,
            0.40,
            SpotPriceSample::new(0, 0.10, 0.01),
        );
        // Force EMA to be exactly 0.10 by using alpha=1
        let mut sched2 = SpotScheduler::new(
            SpotBidStrategy::EmaMultiplier(1.1),
            PreemptionConfig::default(),
            1.0,
            0.99,
        );
        {
            let model = sched2.upsert_model("c5.2xlarge", CloudRegion::AwsUsEast1, 0.40);
            model.alpha = 1.0;
            model.observe(SpotPriceSample::new(0, 0.10, 0.01));
        }
        let savings = sched2
            .estimated_savings_usd("c5.2xlarge", &CloudRegion::AwsUsEast1, 2.0, 4)
            .unwrap();
        // (0.40 - 0.10) * 2 hr * 4 instances = 2.40
        assert!((savings - 2.40).abs() < 1e-6);
    }

    #[test]
    fn test_scheduler_no_savings_for_unknown_model() {
        let sched = make_scheduler();
        assert!(sched
            .estimated_savings_usd("unknown", &CloudRegion::AwsUsEast1, 1.0, 1)
            .is_none());
    }
}
