#![allow(dead_code)]
//! Multi-region cloud deployment configuration and routing.

use std::collections::HashMap;

/// Geographic continent grouping for cloud regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Continent {
    NorthAmerica,
    SouthAmerica,
    Europe,
    AsiaPacific,
    MiddleEast,
    Africa,
}

/// A named cloud region with geographic metadata.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CloudRegion {
    /// AWS us-east-1
    UsEast1,
    /// AWS us-west-2
    UsWest2,
    /// AWS eu-west-1
    EuWest1,
    /// AWS ap-southeast-1
    ApSoutheast1,
    /// AWS sa-east-1
    SaEast1,
    /// AWS me-south-1
    MeSouth1,
    /// AWS af-south-1
    AfSouth1,
    /// Custom region
    Custom(String),
}

impl CloudRegion {
    /// Returns the continent this region belongs to.
    pub fn continent(&self) -> Continent {
        match self {
            Self::UsEast1 | Self::UsWest2 => Continent::NorthAmerica,
            Self::SaEast1 => Continent::SouthAmerica,
            Self::EuWest1 => Continent::Europe,
            Self::ApSoutheast1 => Continent::AsiaPacific,
            Self::MeSouth1 => Continent::MiddleEast,
            Self::AfSouth1 => Continent::Africa,
            Self::Custom(_) => Continent::NorthAmerica,
        }
    }

    /// Returns the region identifier string.
    pub fn id(&self) -> &str {
        match self {
            Self::UsEast1 => "us-east-1",
            Self::UsWest2 => "us-west-2",
            Self::EuWest1 => "eu-west-1",
            Self::ApSoutheast1 => "ap-southeast-1",
            Self::SaEast1 => "sa-east-1",
            Self::MeSouth1 => "me-south-1",
            Self::AfSouth1 => "af-south-1",
            Self::Custom(s) => s.as_str(),
        }
    }
}

/// Health status of a cloud region.
#[derive(Debug, Clone)]
pub struct RegionHealth {
    pub region: CloudRegion,
    pub latency_ms: u32,
    pub error_rate_pct: f32,
    pub available: bool,
}

impl RegionHealth {
    /// Create a new `RegionHealth` record.
    pub fn new(region: CloudRegion, latency_ms: u32, error_rate_pct: f32, available: bool) -> Self {
        Self {
            region,
            latency_ms,
            error_rate_pct,
            available,
        }
    }

    /// Returns `true` if this region should be considered degraded.
    ///
    /// A region is degraded when it is unavailable, has an error rate above 5 %,
    /// or has a latency above 500 ms.
    pub fn is_degraded(&self) -> bool {
        !self.available || self.error_rate_pct > 5.0 || self.latency_ms > 500
    }
}

/// Configuration for a multi-region deployment.
#[derive(Debug, Clone)]
pub struct MultiRegionConfig {
    pub primary_region: CloudRegion,
    pub secondary_regions: Vec<CloudRegion>,
    /// Prefer the region on the same continent as the client when possible.
    pub prefer_same_continent: bool,
    /// Maximum acceptable latency in milliseconds before failing over.
    pub max_latency_ms: u32,
}

impl MultiRegionConfig {
    /// Create a new `MultiRegionConfig`.
    pub fn new(primary_region: CloudRegion) -> Self {
        Self {
            primary_region,
            secondary_regions: Vec::new(),
            prefer_same_continent: true,
            max_latency_ms: 200,
        }
    }

    /// Returns the configured primary region.
    pub fn primary_region(&self) -> &CloudRegion {
        &self.primary_region
    }

    /// Add a secondary region.
    pub fn with_secondary(mut self, region: CloudRegion) -> Self {
        self.secondary_regions.push(region);
        self
    }
}

/// Routes incoming requests to the most suitable cloud region.
#[derive(Debug, Default)]
pub struct MultiRegionRouter {
    health: HashMap<String, RegionHealth>,
}

impl MultiRegionRouter {
    /// Create a new router with no health data.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update health data for a region.
    pub fn update_health(&mut self, health: RegionHealth) {
        self.health.insert(health.region.id().to_string(), health);
    }

    /// Select the best available region from a config.
    ///
    /// Returns `None` if all regions are degraded.
    pub fn select_region<'a>(&self, config: &'a MultiRegionConfig) -> Option<&'a CloudRegion> {
        // Try primary first.
        let primary_id = config.primary_region.id();
        if let Some(h) = self.health.get(primary_id) {
            if !h.is_degraded() {
                return Some(&config.primary_region);
            }
        } else {
            // No health data means assume healthy.
            return Some(&config.primary_region);
        }

        // Fall back to secondaries.
        for region in &config.secondary_regions {
            let id = region.id();
            if let Some(h) = self.health.get(id) {
                if !h.is_degraded() {
                    return Some(region);
                }
            } else {
                return Some(region);
            }
        }

        None
    }

    /// Returns the number of healthy regions tracked.
    #[allow(clippy::cast_precision_loss)]
    pub fn healthy_count(&self) -> usize {
        self.health.values().filter(|h| !h.is_degraded()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_us_east1_continent() {
        assert_eq!(CloudRegion::UsEast1.continent(), Continent::NorthAmerica);
    }

    #[test]
    fn test_eu_west1_continent() {
        assert_eq!(CloudRegion::EuWest1.continent(), Continent::Europe);
    }

    #[test]
    fn test_sa_east1_continent() {
        assert_eq!(CloudRegion::SaEast1.continent(), Continent::SouthAmerica);
    }

    #[test]
    fn test_ap_southeast1_continent() {
        assert_eq!(
            CloudRegion::ApSoutheast1.continent(),
            Continent::AsiaPacific
        );
    }

    #[test]
    fn test_region_id_strings() {
        assert_eq!(CloudRegion::UsEast1.id(), "us-east-1");
        assert_eq!(CloudRegion::UsWest2.id(), "us-west-2");
        assert_eq!(CloudRegion::EuWest1.id(), "eu-west-1");
    }

    #[test]
    fn test_custom_region_id() {
        let r = CloudRegion::Custom("custom-1".to_string());
        assert_eq!(r.id(), "custom-1");
    }

    #[test]
    fn test_region_health_not_degraded() {
        let h = RegionHealth::new(CloudRegion::UsEast1, 50, 0.5, true);
        assert!(!h.is_degraded());
    }

    #[test]
    fn test_region_health_unavailable_is_degraded() {
        let h = RegionHealth::new(CloudRegion::UsEast1, 50, 0.0, false);
        assert!(h.is_degraded());
    }

    #[test]
    fn test_region_health_high_error_rate_degraded() {
        let h = RegionHealth::new(CloudRegion::UsEast1, 50, 10.0, true);
        assert!(h.is_degraded());
    }

    #[test]
    fn test_region_health_high_latency_degraded() {
        let h = RegionHealth::new(CloudRegion::UsEast1, 600, 0.0, true);
        assert!(h.is_degraded());
    }

    #[test]
    fn test_multi_region_config_primary() {
        let cfg = MultiRegionConfig::new(CloudRegion::UsEast1);
        assert_eq!(cfg.primary_region(), &CloudRegion::UsEast1);
    }

    #[test]
    fn test_multi_region_config_with_secondary() {
        let cfg = MultiRegionConfig::new(CloudRegion::UsEast1).with_secondary(CloudRegion::EuWest1);
        assert_eq!(cfg.secondary_regions.len(), 1);
    }

    #[test]
    fn test_router_selects_primary_when_healthy() {
        let cfg = MultiRegionConfig::new(CloudRegion::UsEast1).with_secondary(CloudRegion::EuWest1);
        let mut router = MultiRegionRouter::new();
        router.update_health(RegionHealth::new(CloudRegion::UsEast1, 30, 0.0, true));
        let selected = router
            .select_region(&cfg)
            .expect("selected should be valid");
        assert_eq!(selected, &CloudRegion::UsEast1);
    }

    #[test]
    fn test_router_falls_back_to_secondary() {
        let cfg = MultiRegionConfig::new(CloudRegion::UsEast1).with_secondary(CloudRegion::EuWest1);
        let mut router = MultiRegionRouter::new();
        // Primary degraded.
        router.update_health(RegionHealth::new(CloudRegion::UsEast1, 600, 0.0, true));
        // Secondary healthy.
        router.update_health(RegionHealth::new(CloudRegion::EuWest1, 80, 0.0, true));
        let selected = router
            .select_region(&cfg)
            .expect("selected should be valid");
        assert_eq!(selected, &CloudRegion::EuWest1);
    }

    #[test]
    fn test_router_returns_none_all_degraded() {
        let cfg = MultiRegionConfig::new(CloudRegion::UsEast1).with_secondary(CloudRegion::EuWest1);
        let mut router = MultiRegionRouter::new();
        router.update_health(RegionHealth::new(CloudRegion::UsEast1, 600, 0.0, true));
        router.update_health(RegionHealth::new(CloudRegion::EuWest1, 600, 0.0, true));
        assert!(router.select_region(&cfg).is_none());
    }

    #[test]
    fn test_router_healthy_count() {
        let mut router = MultiRegionRouter::new();
        router.update_health(RegionHealth::new(CloudRegion::UsEast1, 30, 0.0, true));
        router.update_health(RegionHealth::new(CloudRegion::EuWest1, 600, 0.0, true));
        assert_eq!(router.healthy_count(), 1);
    }

    #[test]
    fn test_router_no_health_data_selects_primary() {
        let cfg = MultiRegionConfig::new(CloudRegion::UsEast1);
        let router = MultiRegionRouter::new();
        let selected = router
            .select_region(&cfg)
            .expect("selected should be valid");
        assert_eq!(selected, &CloudRegion::UsEast1);
    }
}
