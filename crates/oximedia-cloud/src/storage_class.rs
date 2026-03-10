//! Storage class configuration and recommendation for cloud object stores.

#![allow(dead_code)]

/// Access-frequency tier used to select an optimal storage class.
/// Named `StorageTierLevel` to avoid collision with `cost::StorageTier`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StorageTierLevel {
    /// Frequently accessed data — lowest latency, highest cost.
    Hot,
    /// Infrequently accessed data — moderate latency and cost.
    Warm,
    /// Rarely accessed data — higher latency, lower cost.
    Cold,
    /// Deep archive — very long retrieval time, very low cost.
    DeepArchive,
}

impl StorageTierLevel {
    /// Typical data-retrieval latency in milliseconds.
    pub fn retrieval_ms(&self) -> u64 {
        match self {
            StorageTierLevel::Hot => 10,
            StorageTierLevel::Warm => 3_000,
            StorageTierLevel::Cold => 180_000,
            StorageTierLevel::DeepArchive => 43_200_000, // 12 hours
        }
    }

    /// Approximate cost per GB per month in USD.
    #[allow(clippy::cast_precision_loss)]
    pub fn base_cost_per_gb_month_usd(&self) -> f64 {
        match self {
            StorageTierLevel::Hot => 0.023,
            StorageTierLevel::Warm => 0.013,
            StorageTierLevel::Cold => 0.004,
            StorageTierLevel::DeepArchive => 0.001,
        }
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            StorageTierLevel::Hot => "Hot",
            StorageTierLevel::Warm => "Warm",
            StorageTierLevel::Cold => "Cold",
            StorageTierLevel::DeepArchive => "Deep Archive",
        }
    }
}

/// Configuration for a named storage class in a cloud provider.
#[derive(Debug, Clone)]
pub struct StorageClassConfig {
    /// Identifier (e.g. "S3-STANDARD", "GCS-NEARLINE").
    pub name: String,
    /// Underlying access tier.
    pub tier: StorageTierLevel,
    /// Minimum storage duration in days (early-deletion fees apply before this).
    pub min_storage_days: u32,
    /// Whether retrieval happens within milliseconds (instant-access tier).
    pub instant_access: bool,
}

impl StorageClassConfig {
    /// Create a new storage class configuration.
    pub fn new(
        name: impl Into<String>,
        tier: StorageTierLevel,
        min_storage_days: u32,
        instant_access: bool,
    ) -> Self {
        Self {
            name: name.into(),
            tier,
            min_storage_days,
            instant_access,
        }
    }

    /// Returns `true` when this class provides instant (sub-second) retrieval.
    pub fn is_instant(&self) -> bool {
        self.instant_access
    }

    /// Cost per GB/month in USD for this class.
    pub fn cost_per_gb_month(&self) -> f64 {
        self.tier.base_cost_per_gb_month_usd()
    }
}

/// Manages a catalogue of `StorageClassConfig` entries and recommends one based on access needs.
#[derive(Debug, Default)]
pub struct StorageClassManager {
    classes: Vec<StorageClassConfig>,
}

impl StorageClassManager {
    /// Create an empty manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a storage class.
    pub fn register(&mut self, config: StorageClassConfig) {
        self.classes.push(config);
    }

    /// Recommend the cheapest class that satisfies the given tier level and instant-access need.
    pub fn recommend(
        &self,
        tier: &StorageTierLevel,
        need_instant: bool,
    ) -> Option<&StorageClassConfig> {
        self.classes
            .iter()
            .filter(|c| &c.tier == tier && (!need_instant || c.instant_access))
            .min_by(|a, b| {
                a.cost_per_gb_month()
                    .partial_cmp(&b.cost_per_gb_month())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Return the cost-per-GB/month for a named class, or `None` if unknown.
    pub fn cost_per_gb_month(&self, name: &str) -> Option<f64> {
        self.classes
            .iter()
            .find(|c| c.name == name)
            .map(|c| c.cost_per_gb_month())
    }

    /// Number of registered classes.
    pub fn len(&self) -> usize {
        self.classes.len()
    }

    /// Returns `true` when no classes are registered.
    pub fn is_empty(&self) -> bool {
        self.classes.is_empty()
    }

    /// All classes for a given tier.
    pub fn classes_for_tier(&self, tier: &StorageTierLevel) -> Vec<&StorageClassConfig> {
        self.classes.iter().filter(|c| &c.tier == tier).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn standard_class() -> StorageClassConfig {
        StorageClassConfig::new("S3-STANDARD", StorageTierLevel::Hot, 0, true)
    }

    fn ia_class() -> StorageClassConfig {
        StorageClassConfig::new("S3-IA", StorageTierLevel::Warm, 30, true)
    }

    fn glacier_class() -> StorageClassConfig {
        StorageClassConfig::new("S3-GLACIER", StorageTierLevel::Cold, 90, false)
    }

    fn deep_class() -> StorageClassConfig {
        StorageClassConfig::new("S3-GLACIER-DA", StorageTierLevel::DeepArchive, 180, false)
    }

    #[test]
    fn test_hot_retrieval_ms() {
        assert_eq!(StorageTierLevel::Hot.retrieval_ms(), 10);
    }

    #[test]
    fn test_deep_archive_retrieval_ms() {
        assert_eq!(StorageTierLevel::DeepArchive.retrieval_ms(), 43_200_000);
    }

    #[test]
    fn test_labels() {
        assert_eq!(StorageTierLevel::Hot.label(), "Hot");
        assert_eq!(StorageTierLevel::Warm.label(), "Warm");
        assert_eq!(StorageTierLevel::Cold.label(), "Cold");
        assert_eq!(StorageTierLevel::DeepArchive.label(), "Deep Archive");
    }

    #[test]
    fn test_standard_is_instant() {
        assert!(standard_class().is_instant());
    }

    #[test]
    fn test_glacier_is_not_instant() {
        assert!(!glacier_class().is_instant());
    }

    #[test]
    fn test_cost_per_gb_month_hot() {
        let c = standard_class();
        assert!((c.cost_per_gb_month() - 0.023).abs() < 1e-9);
    }

    #[test]
    fn test_cost_per_gb_month_deep() {
        let c = deep_class();
        assert!((c.cost_per_gb_month() - 0.001).abs() < 1e-9);
    }

    #[test]
    fn test_manager_register_and_len() {
        let mut mgr = StorageClassManager::new();
        mgr.register(standard_class());
        mgr.register(ia_class());
        assert_eq!(mgr.len(), 2);
    }

    #[test]
    fn test_manager_recommend_hot() {
        let mut mgr = StorageClassManager::new();
        mgr.register(standard_class());
        mgr.register(glacier_class());
        let rec = mgr.recommend(&StorageTierLevel::Hot, false);
        assert!(rec.is_some());
        assert_eq!(rec.expect("test expectation failed").name, "S3-STANDARD");
    }

    #[test]
    fn test_manager_recommend_cold_instant_none() {
        let mut mgr = StorageClassManager::new();
        mgr.register(glacier_class());
        // glacier is not instant
        let rec = mgr.recommend(&StorageTierLevel::Cold, true);
        assert!(rec.is_none());
    }

    #[test]
    fn test_manager_cost_per_gb_month_named() {
        let mut mgr = StorageClassManager::new();
        mgr.register(ia_class());
        let cost = mgr.cost_per_gb_month("S3-IA");
        assert!(cost.is_some());
        assert!((cost.expect("test expectation failed") - 0.013).abs() < 1e-9);
    }

    #[test]
    fn test_manager_cost_per_gb_month_unknown() {
        let mgr = StorageClassManager::new();
        assert!(mgr.cost_per_gb_month("NO-SUCH-CLASS").is_none());
    }

    #[test]
    fn test_classes_for_tier() {
        let mut mgr = StorageClassManager::new();
        mgr.register(standard_class());
        mgr.register(ia_class());
        mgr.register(glacier_class());
        let hot = mgr.classes_for_tier(&StorageTierLevel::Hot);
        assert_eq!(hot.len(), 1);
    }
}
