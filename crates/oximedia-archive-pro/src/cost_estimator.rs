//! Preservation cost estimator for long-term digital archiving.
//!
//! This module calculates projected costs for preserving digital media over time,
//! incorporating storage costs, format migration, fixity checking operations,
//! replication expenses, and staff time. It models cost trajectories across
//! multiple time horizons (1, 5, 10, 25, 50 years).
//!
//! All monetary values are represented in US dollars (f64) and are intended
//! for planning and budgeting purposes only — not as binding financial advice.

use std::collections::HashMap;

/// Storage tier with associated cost characteristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageTier {
    /// Hot storage: immediately accessible (e.g., SSD arrays).
    Hot,
    /// Warm storage: accessible within minutes (e.g., spinning disk).
    Warm,
    /// Cold storage: accessible within hours (e.g., tape, cloud glacier).
    Cold,
    /// Deep archive: accessible within days (e.g., offline tape vaults).
    DeepArchive,
}

impl StorageTier {
    /// Returns the base storage cost in USD per GB per year.
    #[must_use]
    pub const fn base_cost_usd_per_gb_year(&self) -> f64 {
        match self {
            Self::Hot => 2.40,       // ~$0.20/GB/month
            Self::Warm => 0.84,      // ~$0.07/GB/month
            Self::Cold => 0.12,      // ~$0.01/GB/month
            Self::DeepArchive => 0.024, // ~$0.002/GB/month
        }
    }

    /// Returns the retrieval cost in USD per GB.
    #[must_use]
    pub const fn retrieval_cost_usd_per_gb(&self) -> f64 {
        match self {
            Self::Hot => 0.0,
            Self::Warm => 0.01,
            Self::Cold => 0.03,
            Self::DeepArchive => 0.10,
        }
    }

    /// Returns the human-readable name of the tier.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Hot => "Hot",
            Self::Warm => "Warm",
            Self::Cold => "Cold",
            Self::DeepArchive => "Deep Archive",
        }
    }

    /// Returns the expected annual storage cost reduction rate (0.0–1.0).
    ///
    /// Storage costs historically decrease roughly 20% per year (Kryder's Law).
    #[must_use]
    pub const fn annual_cost_decline_rate(&self) -> f64 {
        0.15 // 15% per year (conservative Kryder's law estimate)
    }
}

/// Format migration cost parameters.
#[derive(Debug, Clone)]
pub struct MigrationCostModel {
    /// Average hours of staff time per TB to migrate.
    pub staff_hours_per_tb: f64,
    /// Cost of compute resources per TB migrated (USD).
    pub compute_cost_per_tb: f64,
    /// Expected migration frequency in years.
    pub migration_interval_years: u32,
    /// Hourly rate for preservation staff (USD).
    pub staff_hourly_rate: f64,
}

impl MigrationCostModel {
    /// Creates a default migration cost model with reasonable assumptions.
    #[must_use]
    pub fn default_model() -> Self {
        Self {
            staff_hours_per_tb: 2.0,
            compute_cost_per_tb: 5.0,
            migration_interval_years: 10,
            staff_hourly_rate: 75.0,
        }
    }

    /// Computes the per-TB migration cost (staff + compute).
    #[must_use]
    pub fn cost_per_tb(&self) -> f64 {
        self.staff_hours_per_tb * self.staff_hourly_rate + self.compute_cost_per_tb
    }
}

impl Default for MigrationCostModel {
    fn default() -> Self {
        Self::default_model()
    }
}

/// Fixity checking cost parameters.
#[derive(Debug, Clone)]
pub struct FixityCostModel {
    /// Number of fixity checks per year per object.
    pub checks_per_year: u32,
    /// Compute cost per check per GB (USD).
    pub compute_cost_per_gb_check: f64,
    /// Staff time per 1000 objects per check cycle (hours).
    pub staff_hours_per_1000_objects: f64,
    /// Hourly staff rate (USD).
    pub staff_hourly_rate: f64,
}

impl FixityCostModel {
    /// Creates a default fixity cost model.
    #[must_use]
    pub fn default_model() -> Self {
        Self {
            checks_per_year: 2,
            compute_cost_per_gb_check: 0.001,
            staff_hours_per_1000_objects: 1.0,
            staff_hourly_rate: 75.0,
        }
    }

    /// Computes the annual fixity cost for a given collection.
    #[must_use]
    pub fn annual_cost(&self, total_gb: f64, total_objects: u32) -> f64 {
        let compute =
            total_gb * self.compute_cost_per_gb_check * f64::from(self.checks_per_year);
        let staff = (f64::from(total_objects) / 1000.0)
            * self.staff_hours_per_1000_objects
            * self.staff_hourly_rate
            * f64::from(self.checks_per_year);
        compute + staff
    }
}

impl Default for FixityCostModel {
    fn default() -> Self {
        Self::default_model()
    }
}

/// Description of a collection to estimate costs for.
#[derive(Debug, Clone)]
pub struct CollectionDescriptor {
    /// Human-readable name.
    pub name: String,
    /// Total size in gigabytes.
    pub total_gb: f64,
    /// Number of discrete objects.
    pub total_objects: u32,
    /// Expected annual growth rate (fraction, e.g. 0.05 = 5%).
    pub annual_growth_rate: f64,
    /// Number of replication copies to maintain.
    pub replication_copies: u32,
    /// Primary storage tier.
    pub storage_tier: StorageTier,
}

impl CollectionDescriptor {
    /// Creates a new collection descriptor.
    #[must_use]
    pub fn new(name: impl Into<String>, total_gb: f64, total_objects: u32) -> Self {
        Self {
            name: name.into(),
            total_gb,
            total_objects,
            annual_growth_rate: 0.05,
            replication_copies: 3,
            storage_tier: StorageTier::Cold,
        }
    }

    /// Sets the annual growth rate.
    #[must_use]
    pub fn with_growth_rate(mut self, rate: f64) -> Self {
        self.annual_growth_rate = rate.max(0.0);
        self
    }

    /// Sets the replication copies.
    #[must_use]
    pub fn with_replication(mut self, copies: u32) -> Self {
        self.replication_copies = copies.max(1);
        self
    }

    /// Sets the storage tier.
    #[must_use]
    pub fn with_tier(mut self, tier: StorageTier) -> Self {
        self.storage_tier = tier;
        self
    }

    /// Projected total size in GB after `years` years of growth.
    #[must_use]
    pub fn projected_gb(&self, years: u32) -> f64 {
        self.total_gb * (1.0 + self.annual_growth_rate).powi(years as i32)
    }

    /// Projected object count after `years` years.
    #[must_use]
    pub fn projected_objects(&self, years: u32) -> u32 {
        let factor = (1.0 + self.annual_growth_rate).powi(years as i32);
        (f64::from(self.total_objects) * factor) as u32
    }
}

/// Annual cost breakdown for a single year.
#[derive(Debug, Clone, Default)]
pub struct YearlyCostBreakdown {
    /// Year number (1-based from start of preservation).
    pub year: u32,
    /// Storage cost (USD).
    pub storage_cost: f64,
    /// Migration cost for this year (USD, zero in non-migration years).
    pub migration_cost: f64,
    /// Fixity checking cost (USD).
    pub fixity_cost: f64,
    /// Staff overhead cost (USD).
    pub staff_overhead_cost: f64,
    /// Total cost for this year (USD).
    pub total_cost: f64,
}

impl YearlyCostBreakdown {
    /// Computes the total from individual components.
    pub fn recompute_total(&mut self) {
        self.total_cost =
            self.storage_cost + self.migration_cost + self.fixity_cost + self.staff_overhead_cost;
    }
}

/// A multi-year cost projection.
#[derive(Debug, Clone)]
pub struct CostProjection {
    /// Name of the collection.
    pub collection_name: String,
    /// Per-year breakdowns.
    pub yearly: Vec<YearlyCostBreakdown>,
    /// Cumulative totals by horizon.
    pub cumulative: HashMap<u32, f64>,
    /// Net Present Value discount rate used (0.0 = no discounting).
    pub discount_rate: f64,
    /// Total nominal cost over the full projection period (USD).
    pub total_nominal_cost: f64,
    /// Total NPV-adjusted cost over the projection period (USD).
    pub total_npv_cost: f64,
}

impl CostProjection {
    /// Returns the projected cost for a given year (1-based).
    #[must_use]
    pub fn year_cost(&self, year: u32) -> Option<f64> {
        self.yearly
            .iter()
            .find(|y| y.year == year)
            .map(|y| y.total_cost)
    }

    /// Returns the cumulative cost through a given year.
    #[must_use]
    pub fn cumulative_through(&self, year: u32) -> f64 {
        self.cumulative.get(&year).copied().unwrap_or(0.0)
    }

    /// Finds the year with the highest single-year cost.
    #[must_use]
    pub fn peak_cost_year(&self) -> Option<u32> {
        self.yearly
            .iter()
            .max_by(|a, b| a.total_cost.partial_cmp(&b.total_cost).unwrap_or(std::cmp::Ordering::Equal))
            .map(|y| y.year)
    }
}

/// Configuration for the cost estimation engine.
#[derive(Debug, Clone)]
pub struct CostEstimatorConfig {
    /// Projection horizon in years.
    pub horizon_years: u32,
    /// Net Present Value discount rate (e.g., 0.03 = 3%).
    pub discount_rate: f64,
    /// Annual staff overhead as a fraction of direct costs (e.g., 0.20 = 20%).
    pub staff_overhead_fraction: f64,
    /// Migration cost model.
    pub migration: MigrationCostModel,
    /// Fixity cost model.
    pub fixity: FixityCostModel,
}

impl CostEstimatorConfig {
    /// Creates a default configuration for a 25-year projection.
    #[must_use]
    pub fn default_25_year() -> Self {
        Self {
            horizon_years: 25,
            discount_rate: 0.03,
            staff_overhead_fraction: 0.20,
            migration: MigrationCostModel::default_model(),
            fixity: FixityCostModel::default_model(),
        }
    }
}

impl Default for CostEstimatorConfig {
    fn default() -> Self {
        Self::default_25_year()
    }
}

/// The preservation cost estimator.
#[derive(Debug, Clone, Default)]
pub struct PreservationCostEstimator {
    config: CostEstimatorConfig,
}

impl PreservationCostEstimator {
    /// Creates a new estimator with the given configuration.
    #[must_use]
    pub fn new(config: CostEstimatorConfig) -> Self {
        Self { config }
    }

    /// Produces a multi-year cost projection for the described collection.
    #[must_use]
    pub fn project(&self, collection: &CollectionDescriptor) -> CostProjection {
        let horizon = self.config.horizon_years;
        let mut yearly = Vec::with_capacity(horizon as usize);
        let mut cumulative = HashMap::new();
        let mut running_total = 0.0_f64;
        let mut running_npv = 0.0_f64;

        for year in 1..=horizon {
            let projected_gb = collection.projected_gb(year);
            let projected_objects = collection.projected_objects(year);
            let effective_gb = projected_gb * f64::from(collection.replication_copies);

            // Storage cost with Kryder's law discount applied year-over-year
            let cost_decline_factor =
                (1.0 - collection.storage_tier.annual_cost_decline_rate()).powi(year as i32);
            let base_storage_rate = collection.storage_tier.base_cost_usd_per_gb_year();
            let storage_cost = effective_gb * base_storage_rate * cost_decline_factor;

            // Migration cost (only in migration years)
            let migration_cost =
                if year % self.config.migration.migration_interval_years == 0 {
                    let tb = projected_gb / 1024.0;
                    tb * self.config.migration.cost_per_tb()
                } else {
                    0.0
                };

            // Fixity cost
            let fixity_cost = self
                .config
                .fixity
                .annual_cost(projected_gb, projected_objects);

            // Staff overhead
            let direct_costs = storage_cost + migration_cost + fixity_cost;
            let staff_overhead_cost = direct_costs * self.config.staff_overhead_fraction;

            let total_cost = direct_costs + staff_overhead_cost;

            // NPV discounting
            let discount_factor = (1.0 + self.config.discount_rate).powi(year as i32);
            let npv_cost = total_cost / discount_factor;

            running_total += total_cost;
            running_npv += npv_cost;

            let mut breakdown = YearlyCostBreakdown {
                year,
                storage_cost,
                migration_cost,
                fixity_cost,
                staff_overhead_cost,
                total_cost,
            };
            breakdown.recompute_total();

            cumulative.insert(year, running_total);
            yearly.push(breakdown);
        }

        CostProjection {
            collection_name: collection.name.clone(),
            yearly,
            cumulative,
            discount_rate: self.config.discount_rate,
            total_nominal_cost: running_total,
            total_npv_cost: running_npv,
        }
    }

    /// Computes cost breakdowns for multiple storage tiers for comparison.
    #[must_use]
    pub fn compare_tiers(
        &self,
        collection: &CollectionDescriptor,
        tiers: &[StorageTier],
    ) -> HashMap<String, f64> {
        let mut result = HashMap::new();
        for &tier in tiers {
            let mut c = collection.clone();
            c.storage_tier = tier;
            let projection = self.project(&c);
            result.insert(tier.name().to_string(), projection.total_nominal_cost);
        }
        result
    }

    /// Returns a human-readable cost summary string.
    #[must_use]
    pub fn summary_text(&self, projection: &CostProjection) -> String {
        format!(
            "Collection: {}\n\
             Horizon: {} years\n\
             Total nominal cost: ${:.2}\n\
             Total NPV cost ({}% discount): ${:.2}\n\
             Peak cost year: {}\n",
            projection.collection_name,
            self.config.horizon_years,
            projection.total_nominal_cost,
            (self.config.discount_rate * 100.0) as u32,
            projection.total_npv_cost,
            projection
                .peak_cost_year()
                .map(|y| y.to_string())
                .unwrap_or_else(|| "N/A".to_string()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_collection() -> CollectionDescriptor {
        CollectionDescriptor::new("Test Archive", 10_000.0, 50_000)
            .with_growth_rate(0.05)
            .with_replication(3)
            .with_tier(StorageTier::Cold)
    }

    #[test]
    fn test_storage_tier_costs_ordered() {
        // Hot > Warm > Cold > DeepArchive
        assert!(
            StorageTier::Hot.base_cost_usd_per_gb_year()
                > StorageTier::Warm.base_cost_usd_per_gb_year()
        );
        assert!(
            StorageTier::Warm.base_cost_usd_per_gb_year()
                > StorageTier::Cold.base_cost_usd_per_gb_year()
        );
        assert!(
            StorageTier::Cold.base_cost_usd_per_gb_year()
                > StorageTier::DeepArchive.base_cost_usd_per_gb_year()
        );
    }

    #[test]
    fn test_storage_tier_names_not_empty() {
        for tier in [
            StorageTier::Hot,
            StorageTier::Warm,
            StorageTier::Cold,
            StorageTier::DeepArchive,
        ] {
            assert!(!tier.name().is_empty());
        }
    }

    #[test]
    fn test_migration_cost_model_per_tb() {
        let m = MigrationCostModel::default_model();
        let cost = m.cost_per_tb();
        // 2 hours * $75/hr + $5 compute = $155
        assert!((cost - 155.0).abs() < 0.01);
    }

    #[test]
    fn test_fixity_cost_model_annual() {
        let m = FixityCostModel::default_model();
        let cost = m.annual_cost(1000.0, 5000);
        assert!(cost > 0.0, "annual fixity cost should be positive");
    }

    #[test]
    fn test_collection_descriptor_projected_gb() {
        let c = CollectionDescriptor::new("Test", 1000.0, 100).with_growth_rate(0.10);
        let y1 = c.projected_gb(1);
        assert!((y1 - 1100.0).abs() < 0.01);
        let y5 = c.projected_gb(5);
        assert!(y5 > y1, "year 5 should be bigger than year 1");
    }

    #[test]
    fn test_collection_descriptor_projected_objects() {
        let c = CollectionDescriptor::new("Test", 1000.0, 1000).with_growth_rate(0.0);
        assert_eq!(c.projected_objects(5), 1000);
    }

    #[test]
    fn test_projection_years_count() {
        let mut config = CostEstimatorConfig::default_25_year();
        config.horizon_years = 10;
        let estimator = PreservationCostEstimator::new(config);
        let projection = estimator.project(&default_collection());
        assert_eq!(projection.yearly.len(), 10);
    }

    #[test]
    fn test_projection_costs_positive() {
        let estimator = PreservationCostEstimator::default();
        let projection = estimator.project(&default_collection());
        for year_cost in &projection.yearly {
            assert!(year_cost.total_cost > 0.0, "year {} cost must be positive", year_cost.year);
        }
    }

    #[test]
    fn test_cumulative_is_monotone() {
        let estimator = PreservationCostEstimator::default();
        let projection = estimator.project(&default_collection());
        let mut prev = 0.0_f64;
        for year in 1..=25 {
            let cum = projection.cumulative_through(year);
            assert!(cum >= prev, "cumulative should be non-decreasing at year {year}");
            prev = cum;
        }
    }

    #[test]
    fn test_migration_cost_appears_in_migration_year() {
        let mut config = CostEstimatorConfig::default_25_year();
        config.migration.migration_interval_years = 5;
        config.horizon_years = 10;
        let estimator = PreservationCostEstimator::new(config);
        let projection = estimator.project(&default_collection());

        let year5 = projection.yearly.iter().find(|y| y.year == 5).expect("year 5");
        let year6 = projection.yearly.iter().find(|y| y.year == 6).expect("year 6");
        assert!(year5.migration_cost > 0.0, "migration cost should be non-zero in year 5");
        assert_eq!(year6.migration_cost, 0.0, "migration cost should be zero in year 6");
    }

    #[test]
    fn test_npv_less_than_nominal() {
        let estimator = PreservationCostEstimator::default();
        let projection = estimator.project(&default_collection());
        assert!(
            projection.total_npv_cost < projection.total_nominal_cost,
            "NPV should be less than nominal with positive discount rate"
        );
    }

    #[test]
    fn test_compare_tiers() {
        let estimator = PreservationCostEstimator::default();
        let collection = default_collection();
        let comparison = estimator.compare_tiers(
            &collection,
            &[StorageTier::Hot, StorageTier::Warm, StorageTier::Cold, StorageTier::DeepArchive],
        );
        assert_eq!(comparison.len(), 4);
        let hot_cost = comparison["Hot"];
        let cold_cost = comparison["Cold"];
        assert!(hot_cost > cold_cost, "hot storage should cost more than cold");
    }

    #[test]
    fn test_summary_text_contains_collection_name() {
        let estimator = PreservationCostEstimator::default();
        let projection = estimator.project(&default_collection());
        let text = estimator.summary_text(&projection);
        assert!(text.contains("Test Archive"));
        assert!(text.contains("25 years"));
    }

    #[test]
    fn test_peak_cost_year_exists() {
        let estimator = PreservationCostEstimator::default();
        let projection = estimator.project(&default_collection());
        let peak = projection.peak_cost_year();
        assert!(peak.is_some());
        let peak_year = peak.expect("peak year");
        assert!(peak_year >= 1 && peak_year <= 25);
    }

    #[test]
    fn test_year_cost_lookup() {
        let estimator = PreservationCostEstimator::default();
        let projection = estimator.project(&default_collection());
        assert!(projection.year_cost(1).is_some());
        assert!(projection.year_cost(0).is_none());
        assert!(projection.year_cost(26).is_none());
    }
}
