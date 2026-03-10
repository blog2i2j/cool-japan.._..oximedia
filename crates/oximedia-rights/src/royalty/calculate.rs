//! Royalty calculation engine
//!
//! Provides per-play, percentage, flat-fee, and tiered royalty calculation
//! methods with full usage-log aggregation.

#![allow(clippy::cast_precision_loss)]

use crate::{database::RightsDatabase, usage::UsageLog, Result};
use chrono::{DateTime, Utc};

// ── Tier definition ──────────────────────────────────────────────────────────

/// A single tier in a tiered royalty schedule.
///
/// Tiers are evaluated in order: the rate applies to plays falling within
/// `[threshold_plays, next_tier_threshold)`.
#[derive(Debug, Clone)]
pub struct RoyaltyTier {
    /// Minimum play count for this tier (inclusive).
    pub threshold_plays: u64,
    /// Rate applied within this tier (interpretation depends on method).
    pub rate: f64,
}

impl RoyaltyTier {
    /// Create a new royalty tier.
    pub fn new(threshold_plays: u64, rate: f64) -> Self {
        Self {
            threshold_plays,
            rate,
        }
    }
}

// ── Royalty method ───────────────────────────────────────────────────────────

/// Royalty calculation method
#[derive(Debug, Clone)]
pub enum RoyaltyMethod {
    /// Fixed amount per use
    PerUse(f64),
    /// Percentage of revenue
    Percentage(f64),
    /// Flat fee regardless of usage
    FlatFee(f64),
    /// Tiered per-play rates.
    ///
    /// Each tier specifies a threshold (cumulative play count) and a per-play
    /// rate.  Plays are distributed across tiers in ascending order:
    ///
    /// ```text
    /// Tier 0: 0–999 plays   → $0.10 / play
    /// Tier 1: 1000–9999     → $0.05 / play
    /// Tier 2: 10000+        → $0.02 / play
    /// ```
    ///
    /// Tiers must be sorted by `threshold_plays` (ascending).
    TieredPerPlay(Vec<RoyaltyTier>),
    /// Tiered percentage of revenue.
    ///
    /// Works like `TieredPerPlay` but the rate is a percentage applied to
    /// per-play revenue rather than a flat per-play amount.
    TieredPercentage(Vec<RoyaltyTier>),
}

// ── Calculator ───────────────────────────────────────────────────────────────

/// Royalty calculator
pub struct RoyaltyCalculator {
    method: RoyaltyMethod,
}

impl RoyaltyCalculator {
    /// Create a new royalty calculator
    pub fn new(method: RoyaltyMethod) -> Self {
        Self { method }
    }

    /// Get a reference to the calculation method.
    pub fn method(&self) -> &RoyaltyMethod {
        &self.method
    }

    /// Calculate royalties for a period (async, database-backed).
    ///
    /// Currently returns the base method result; a production implementation
    /// would query usage logs from `_db` for the given grant and period.
    pub async fn calculate(
        &self,
        _db: &RightsDatabase,
        _grant_id: &str,
        _start: DateTime<Utc>,
        _end: DateTime<Utc>,
    ) -> Result<f64> {
        match &self.method {
            RoyaltyMethod::PerUse(amount) => Ok(*amount),
            RoyaltyMethod::Percentage(pct) => Ok(*pct),
            RoyaltyMethod::FlatFee(fee) => Ok(*fee),
            RoyaltyMethod::TieredPerPlay(_) | RoyaltyMethod::TieredPercentage(_) => {
                // Without usage data from DB, return 0
                Ok(0.0)
            }
        }
    }

    /// Calculate from usage logs
    pub fn calculate_from_usage(
        &self,
        usage_logs: &[UsageLog],
        revenue_per_use: Option<f64>,
    ) -> f64 {
        let play_count = usage_logs.len() as u64;
        match &self.method {
            RoyaltyMethod::PerUse(amount) => play_count as f64 * amount,
            RoyaltyMethod::Percentage(pct) => {
                if let Some(revenue) = revenue_per_use {
                    play_count as f64 * revenue * (pct / 100.0)
                } else {
                    0.0
                }
            }
            RoyaltyMethod::FlatFee(fee) => *fee,
            RoyaltyMethod::TieredPerPlay(tiers) => {
                Self::compute_tiered_amount(play_count, tiers, None)
            }
            RoyaltyMethod::TieredPercentage(tiers) => {
                Self::compute_tiered_amount(play_count, tiers, revenue_per_use)
            }
        }
    }

    /// Distribute `total_plays` across sorted tiers and compute the total royalty.
    ///
    /// For `TieredPerPlay` pass `revenue_per_use = None`; the tier rate is the
    /// per-play dollar amount.
    ///
    /// For `TieredPercentage` pass `Some(revenue)` and the tier rate is a
    /// percentage applied to `revenue * plays_in_tier`.
    fn compute_tiered_amount(
        total_plays: u64,
        tiers: &[RoyaltyTier],
        revenue_per_use: Option<f64>,
    ) -> f64 {
        if tiers.is_empty() || total_plays == 0 {
            return 0.0;
        }

        let mut total = 0.0f64;
        let mut remaining = total_plays;

        for (i, tier) in tiers.iter().enumerate() {
            // Upper bound is the next tier's threshold (or infinity).
            let upper = tiers
                .get(i + 1)
                .map(|t| t.threshold_plays)
                .unwrap_or(u64::MAX);

            // How many plays fall in *this* tier?
            let tier_width = upper.saturating_sub(tier.threshold_plays);
            let plays_in_tier = remaining.min(tier_width);

            if plays_in_tier == 0 {
                // All plays already above this tier's threshold range
                // but below its start – skip.
                continue;
            }

            match revenue_per_use {
                Some(revenue) => {
                    // tier.rate is a percentage
                    total += plays_in_tier as f64 * revenue * (tier.rate / 100.0);
                }
                None => {
                    // tier.rate is a per-play amount
                    total += plays_in_tier as f64 * tier.rate;
                }
            }

            remaining = remaining.saturating_sub(plays_in_tier);
            if remaining == 0 {
                break;
            }
        }

        total
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_per_use_calculation() {
        let calculator = RoyaltyCalculator::new(RoyaltyMethod::PerUse(10.0));
        let usage_logs = vec![
            UsageLog::new("asset1", crate::rights::UsageType::Commercial, Utc::now()),
            UsageLog::new("asset1", crate::rights::UsageType::Web, Utc::now()),
        ];

        let total = calculator.calculate_from_usage(&usage_logs, None);
        assert_eq!(total, 20.0);
    }

    #[test]
    fn test_percentage_calculation() {
        let calculator = RoyaltyCalculator::new(RoyaltyMethod::Percentage(10.0));
        let usage_logs = vec![UsageLog::new(
            "asset1",
            crate::rights::UsageType::Commercial,
            Utc::now(),
        )];

        let total = calculator.calculate_from_usage(&usage_logs, Some(100.0));
        assert_eq!(total, 10.0);
    }

    #[test]
    fn test_flat_fee() {
        let calculator = RoyaltyCalculator::new(RoyaltyMethod::FlatFee(500.0));
        let usage_logs = vec![
            UsageLog::new("a", crate::rights::UsageType::Commercial, Utc::now()),
            UsageLog::new("a", crate::rights::UsageType::Commercial, Utc::now()),
        ];
        assert_eq!(calculator.calculate_from_usage(&usage_logs, None), 500.0);
    }

    #[test]
    fn test_tiered_per_play_single_tier() {
        let tiers = vec![RoyaltyTier::new(0, 0.10)];
        let calculator = RoyaltyCalculator::new(RoyaltyMethod::TieredPerPlay(tiers));

        let logs: Vec<UsageLog> = (0..100)
            .map(|_| UsageLog::new("a", crate::rights::UsageType::Commercial, Utc::now()))
            .collect();

        let total = calculator.calculate_from_usage(&logs, None);
        assert!((total - 10.0).abs() < 1e-9); // 100 * 0.10
    }

    #[test]
    fn test_tiered_per_play_multiple_tiers() {
        // Tier 0: 0–99  plays  → $0.10 / play
        // Tier 1: 100–999      → $0.05 / play
        // Tier 2: 1000+        → $0.02 / play
        let tiers = vec![
            RoyaltyTier::new(0, 0.10),
            RoyaltyTier::new(100, 0.05),
            RoyaltyTier::new(1000, 0.02),
        ];
        let calculator = RoyaltyCalculator::new(RoyaltyMethod::TieredPerPlay(tiers));

        // 250 plays: 100 @ $0.10 + 150 @ $0.05 = $10 + $7.50 = $17.50
        let logs: Vec<UsageLog> = (0..250)
            .map(|_| UsageLog::new("a", crate::rights::UsageType::Commercial, Utc::now()))
            .collect();

        let total = calculator.calculate_from_usage(&logs, None);
        assert!((total - 17.5).abs() < 1e-9);
    }

    #[test]
    fn test_tiered_per_play_spans_all_tiers() {
        let tiers = vec![
            RoyaltyTier::new(0, 0.10),
            RoyaltyTier::new(100, 0.05),
            RoyaltyTier::new(1000, 0.02),
        ];
        let calculator = RoyaltyCalculator::new(RoyaltyMethod::TieredPerPlay(tiers));

        // 1500 plays: 100@0.10 + 900@0.05 + 500@0.02
        // = $10 + $45 + $10 = $65
        let logs: Vec<UsageLog> = (0..1500)
            .map(|_| UsageLog::new("a", crate::rights::UsageType::Commercial, Utc::now()))
            .collect();

        let total = calculator.calculate_from_usage(&logs, None);
        assert!((total - 65.0).abs() < 1e-9);
    }

    #[test]
    fn test_tiered_percentage() {
        // Tier 0: 0–99 plays   → 10% of revenue per play
        // Tier 1: 100+         → 5% of revenue per play
        let tiers = vec![RoyaltyTier::new(0, 10.0), RoyaltyTier::new(100, 5.0)];
        let calculator = RoyaltyCalculator::new(RoyaltyMethod::TieredPercentage(tiers));

        // 150 plays at $1.00 revenue each
        // 100 @ 10% of $1 = $10.00
        //  50 @  5% of $1 = $2.50
        // Total = $12.50
        let logs: Vec<UsageLog> = (0..150)
            .map(|_| UsageLog::new("a", crate::rights::UsageType::Commercial, Utc::now()))
            .collect();

        let total = calculator.calculate_from_usage(&logs, Some(1.0));
        assert!((total - 12.5).abs() < 1e-9);
    }

    #[test]
    fn test_tiered_empty_logs() {
        let tiers = vec![RoyaltyTier::new(0, 0.10)];
        let calculator = RoyaltyCalculator::new(RoyaltyMethod::TieredPerPlay(tiers));
        assert_eq!(calculator.calculate_from_usage(&[], None), 0.0);
    }

    #[test]
    fn test_tiered_empty_tiers() {
        let calculator = RoyaltyCalculator::new(RoyaltyMethod::TieredPerPlay(vec![]));
        let logs = vec![UsageLog::new(
            "a",
            crate::rights::UsageType::Commercial,
            Utc::now(),
        )];
        assert_eq!(calculator.calculate_from_usage(&logs, None), 0.0);
    }
}
