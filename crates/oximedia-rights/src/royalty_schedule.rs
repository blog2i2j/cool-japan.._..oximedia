//! Royalty payment schedules and distribution split management.
//!
//! This module handles how royalty payments are scheduled over time and how
//! revenue is split among multiple rights holders.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

// ── PaymentFrequency ──────────────────────────────────────────────────────────

/// How often royalty payments are made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaymentFrequency {
    /// Payment made once a month.
    Monthly,
    /// Payment made once a quarter (every 3 months).
    Quarterly,
    /// Payment made twice a year.
    SemiAnnual,
    /// Payment made once a year.
    Annual,
}

impl PaymentFrequency {
    /// Return the number of payment periods per year.
    pub fn periods_per_year(self) -> u32 {
        match self {
            PaymentFrequency::Monthly => 12,
            PaymentFrequency::Quarterly => 4,
            PaymentFrequency::SemiAnnual => 2,
            PaymentFrequency::Annual => 1,
        }
    }
}

// ── PaymentSchedule ───────────────────────────────────────────────────────────

/// Defines when and how much is paid to a single rights holder.
#[derive(Debug, Clone)]
pub struct PaymentSchedule {
    /// Unique schedule identifier.
    pub id: u32,
    /// Rights holder name.
    pub holder_name: String,
    /// Frequency of payments.
    pub frequency: PaymentFrequency,
    /// Base amount paid per period (in currency units).
    pub amount_per_period: f64,
    /// ISO 4217 currency code.
    pub currency: String,
    /// Total number of periods this schedule is active.  `None` means
    /// indefinite.
    pub total_periods: Option<u32>,
}

impl PaymentSchedule {
    /// Create a new `PaymentSchedule`.
    pub fn new(
        id: u32,
        holder_name: impl Into<String>,
        frequency: PaymentFrequency,
        amount_per_period: f64,
        currency: impl Into<String>,
        total_periods: Option<u32>,
    ) -> Self {
        Self {
            id,
            holder_name: holder_name.into(),
            frequency,
            amount_per_period: amount_per_period.max(0.0),
            currency: currency.into(),
            total_periods,
        }
    }

    /// Total payout over the life of this schedule.
    /// Returns `None` for indefinite schedules.
    pub fn total_payout(&self) -> Option<f64> {
        self.total_periods
            .map(|n| n as f64 * self.amount_per_period)
    }

    /// Annual equivalent payment amount.
    pub fn annual_equivalent(&self) -> f64 {
        self.amount_per_period * self.frequency.periods_per_year() as f64
    }
}

// ── DistributionSplit ─────────────────────────────────────────────────────────

/// A single holder's share in a distribution.
#[derive(Debug, Clone)]
pub struct DistributionSplit {
    /// Rights holder name.
    pub holder_name: String,
    /// Share percentage (0.0 – 100.0).
    pub share_pct: f64,
}

impl DistributionSplit {
    /// Create a new `DistributionSplit`, clamping to \[0.0, 100.0\].
    pub fn new(holder_name: impl Into<String>, share_pct: f64) -> Self {
        Self {
            holder_name: holder_name.into(),
            share_pct: share_pct.clamp(0.0, 100.0),
        }
    }
}

// ── DistributionTable ─────────────────────────────────────────────────────────

/// A table mapping rights holders to their percentage shares.
#[derive(Debug, Default)]
pub struct DistributionTable {
    splits: Vec<DistributionSplit>,
}

impl DistributionTable {
    /// Create an empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a split entry.
    pub fn add_split(&mut self, split: DistributionSplit) {
        self.splits.push(split);
    }

    /// Sum of all share percentages.  Should equal 100.0 for a valid table.
    pub fn total_share(&self) -> f64 {
        self.splits.iter().map(|s| s.share_pct).sum()
    }

    /// Return `true` if the total share is approximately 100 %.
    pub fn is_balanced(&self) -> bool {
        (self.total_share() - 100.0).abs() < 0.01
    }

    /// Calculate how much of `total_amount` each holder receives.
    pub fn distribute(&self, total_amount: f64) -> HashMap<&str, f64> {
        self.splits
            .iter()
            .map(|s| (s.holder_name.as_str(), total_amount * s.share_pct / 100.0))
            .collect()
    }

    /// Return the split for a given holder name.
    pub fn find(&self, holder_name: &str) -> Option<&DistributionSplit> {
        self.splits.iter().find(|s| s.holder_name == holder_name)
    }

    /// Number of splits in the table.
    pub fn len(&self) -> usize {
        self.splits.len()
    }

    /// `true` if the table has no splits.
    pub fn is_empty(&self) -> bool {
        self.splits.is_empty()
    }
}

// ── ScheduleRegistry ─────────────────────────────────────────────────────────

/// Central registry that holds all payment schedules for a production.
#[derive(Debug, Default)]
pub struct ScheduleRegistry {
    schedules: HashMap<u32, PaymentSchedule>,
}

impl ScheduleRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a schedule.  Returns `Err` if the ID is already taken.
    pub fn register(&mut self, schedule: PaymentSchedule) -> Result<(), String> {
        if self.schedules.contains_key(&schedule.id) {
            return Err(format!("Schedule id {} already registered", schedule.id));
        }
        self.schedules.insert(schedule.id, schedule);
        Ok(())
    }

    /// Look up a schedule by ID.
    pub fn get(&self, id: u32) -> Option<&PaymentSchedule> {
        self.schedules.get(&id)
    }

    /// Total annual obligation across all registered schedules.
    pub fn total_annual_obligation(&self) -> f64 {
        self.schedules
            .values()
            .map(PaymentSchedule::annual_equivalent)
            .sum()
    }

    /// Number of registered schedules.
    pub fn len(&self) -> usize {
        self.schedules.len()
    }

    /// `true` if no schedules are registered.
    pub fn is_empty(&self) -> bool {
        self.schedules.is_empty()
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn monthly_schedule(id: u32, amount: f64) -> PaymentSchedule {
        PaymentSchedule::new(
            id,
            "Alice",
            PaymentFrequency::Monthly,
            amount,
            "USD",
            Some(12),
        )
    }

    #[test]
    fn test_payment_frequency_periods_per_year() {
        assert_eq!(PaymentFrequency::Monthly.periods_per_year(), 12);
        assert_eq!(PaymentFrequency::Quarterly.periods_per_year(), 4);
        assert_eq!(PaymentFrequency::SemiAnnual.periods_per_year(), 2);
        assert_eq!(PaymentFrequency::Annual.periods_per_year(), 1);
    }

    #[test]
    fn test_payment_schedule_total_payout() {
        let sched = monthly_schedule(1, 100.0);
        assert!(
            (sched
                .total_payout()
                .expect("rights test operation should succeed")
                - 1200.0)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn test_payment_schedule_total_payout_indefinite() {
        let sched = PaymentSchedule::new(1, "Bob", PaymentFrequency::Annual, 500.0, "EUR", None);
        assert!(sched.total_payout().is_none());
    }

    #[test]
    fn test_payment_schedule_annual_equivalent_monthly() {
        let sched = monthly_schedule(1, 100.0);
        assert!((sched.annual_equivalent() - 1200.0).abs() < 1e-9);
    }

    #[test]
    fn test_payment_schedule_annual_equivalent_quarterly() {
        let sched =
            PaymentSchedule::new(1, "Carol", PaymentFrequency::Quarterly, 250.0, "GBP", None);
        assert!((sched.annual_equivalent() - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn test_payment_schedule_amount_clamped_to_zero() {
        let sched = PaymentSchedule::new(1, "Dave", PaymentFrequency::Monthly, -50.0, "USD", None);
        assert_eq!(sched.amount_per_period, 0.0);
    }

    #[test]
    fn test_distribution_split_clamped() {
        let split = DistributionSplit::new("Alice", 150.0);
        assert_eq!(split.share_pct, 100.0);
    }

    #[test]
    fn test_distribution_table_is_balanced() {
        let mut table = DistributionTable::new();
        table.add_split(DistributionSplit::new("Alice", 60.0));
        table.add_split(DistributionSplit::new("Bob", 40.0));
        assert!(table.is_balanced());
    }

    #[test]
    fn test_distribution_table_not_balanced() {
        let mut table = DistributionTable::new();
        table.add_split(DistributionSplit::new("Alice", 60.0));
        assert!(!table.is_balanced());
    }

    #[test]
    fn test_distribution_table_distribute() {
        let mut table = DistributionTable::new();
        table.add_split(DistributionSplit::new("Alice", 75.0));
        table.add_split(DistributionSplit::new("Bob", 25.0));
        let shares = table.distribute(1000.0);
        assert!((shares["Alice"] - 750.0).abs() < 1e-9);
        assert!((shares["Bob"] - 250.0).abs() < 1e-9);
    }

    #[test]
    fn test_distribution_table_find() {
        let mut table = DistributionTable::new();
        table.add_split(DistributionSplit::new("Alice", 50.0));
        assert!(table.find("Alice").is_some());
        assert!(table.find("Charlie").is_none());
    }

    #[test]
    fn test_schedule_registry_register_and_get() {
        let mut reg = ScheduleRegistry::new();
        reg.register(monthly_schedule(1, 200.0))
            .expect("rights test operation should succeed");
        assert!(reg.get(1).is_some());
        assert!(reg.get(99).is_none());
    }

    #[test]
    fn test_schedule_registry_duplicate_rejected() {
        let mut reg = ScheduleRegistry::new();
        reg.register(monthly_schedule(1, 200.0))
            .expect("rights test operation should succeed");
        assert!(reg.register(monthly_schedule(1, 300.0)).is_err());
    }

    #[test]
    fn test_schedule_registry_total_annual_obligation() {
        let mut reg = ScheduleRegistry::new();
        // Monthly 100 * 12 = 1200
        reg.register(monthly_schedule(1, 100.0))
            .expect("rights test operation should succeed");
        // Quarterly 250 * 4 = 1000
        reg.register(PaymentSchedule::new(
            2,
            "Bob",
            PaymentFrequency::Quarterly,
            250.0,
            "USD",
            None,
        ))
        .expect("rights test operation should succeed");
        assert!((reg.total_annual_obligation() - 2200.0).abs() < 1e-9);
    }
}
