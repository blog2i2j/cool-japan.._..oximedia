//! CDN cost analytics.
//!
//! [`CdnCostAnalytics`] records data transfer events per CDN provider and
//! accumulates the total dollar cost based on configurable per-GB pricing.

use std::collections::HashMap;

/// Per-provider transfer record.
#[derive(Debug, Default, Clone)]
struct ProviderStats {
    /// Total bytes transferred.
    total_bytes: u64,
    /// Accumulated dollar cost.
    total_cost_usd: f64,
}

/// CDN cost analytics ledger.
///
/// Records data transfer volumes and computes costs per provider and overall.
///
/// # Example
/// ```
/// use oximedia_cdn::cost::CdnCostAnalytics;
///
/// let mut analytics = CdnCostAnalytics::new();
/// // 1 GiB at $0.08/GB
/// analytics.record_transfer("cloudfront", 1_073_741_824, 0.08);
/// // 512 MiB at $0.06/GB
/// analytics.record_transfer("fastly", 536_870_912, 0.06);
/// let total = analytics.total_cost();
/// assert!(total > 0.0);
/// ```
#[derive(Debug, Default)]
pub struct CdnCostAnalytics {
    providers: HashMap<String, ProviderStats>,
}

const BYTES_PER_GB: f64 = 1_073_741_824.0; // 1 GiB = 2^30 bytes

impl CdnCostAnalytics {
    /// Create an empty cost analytics ledger.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a data transfer event.
    ///
    /// - `cdn` – provider identifier.
    /// - `bytes` – number of bytes transferred.
    /// - `cost_per_gb` – provider's per-GB cost in USD.
    pub fn record_transfer(&mut self, cdn: &str, bytes: u64, cost_per_gb: f64) {
        let cost = (bytes as f64 / BYTES_PER_GB) * cost_per_gb;
        let entry = self.providers.entry(cdn.to_string()).or_default();
        entry.total_bytes += bytes;
        entry.total_cost_usd += cost;
    }

    /// Total cost across all providers in USD.
    pub fn total_cost(&self) -> f64 {
        self.providers.values().map(|s| s.total_cost_usd).sum()
    }

    /// Cost attributed to a specific CDN provider, or `0.0` if unknown.
    pub fn cost_for(&self, cdn: &str) -> f64 {
        self.providers
            .get(cdn)
            .map(|s| s.total_cost_usd)
            .unwrap_or(0.0)
    }

    /// Total bytes transferred across all providers.
    pub fn total_bytes(&self) -> u64 {
        self.providers.values().map(|s| s.total_bytes).sum()
    }

    /// Total bytes transferred for a specific provider.
    pub fn bytes_for(&self, cdn: &str) -> u64 {
        self.providers
            .get(cdn)
            .map(|s| s.total_bytes)
            .unwrap_or(0)
    }

    /// Number of distinct CDN providers tracked.
    pub fn provider_count(&self) -> usize {
        self.providers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-9;

    #[test]
    fn test_empty_cost_is_zero() {
        let analytics = CdnCostAnalytics::new();
        assert!((analytics.total_cost() - 0.0).abs() < EPSILON);
        assert_eq!(analytics.total_bytes(), 0);
    }

    #[test]
    fn test_record_transfer_one_gb() {
        let mut analytics = CdnCostAnalytics::new();
        // Exactly 1 GiB at $0.10/GB → $0.10
        analytics.record_transfer("cf", 1_073_741_824, 0.10);
        assert!((analytics.total_cost() - 0.10).abs() < EPSILON);
        assert!((analytics.cost_for("cf") - 0.10).abs() < EPSILON);
        assert_eq!(analytics.total_bytes(), 1_073_741_824);
    }

    #[test]
    fn test_multiple_providers() {
        let mut analytics = CdnCostAnalytics::new();
        analytics.record_transfer("a", 1_073_741_824, 0.08); // $0.08
        analytics.record_transfer("b", 1_073_741_824, 0.06); // $0.06
        assert!((analytics.total_cost() - 0.14).abs() < EPSILON);
        assert_eq!(analytics.provider_count(), 2);
    }

    #[test]
    fn test_accumulate_same_provider() {
        let mut analytics = CdnCostAnalytics::new();
        analytics.record_transfer("x", 1_073_741_824, 0.05); // $0.05
        analytics.record_transfer("x", 1_073_741_824, 0.05); // $0.05
        assert!((analytics.cost_for("x") - 0.10).abs() < EPSILON);
        assert_eq!(analytics.bytes_for("x"), 2 * 1_073_741_824);
    }

    #[test]
    fn test_unknown_provider_returns_zero() {
        let analytics = CdnCostAnalytics::new();
        assert!((analytics.cost_for("unknown")).abs() < EPSILON);
        assert_eq!(analytics.bytes_for("unknown"), 0);
    }
}
