//! CDN integration and management
//!
//! Provides abstractions for managing Content Delivery Networks across multiple
//! providers including CloudFront, Fastly, Cloudflare, Akamai, and BunnyCDN.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// CDN provider selection
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CdnProvider {
    /// Amazon CloudFront
    Cloudfront,
    /// Fastly CDN
    Fastly,
    /// Cloudflare
    Cloudflare,
    /// Akamai
    Akamai,
    /// BunnyCDN
    BunnyCdn,
    /// Custom provider
    Custom(String),
}

impl CdnProvider {
    /// Human-readable name of the provider
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            CdnProvider::Cloudfront => "Amazon CloudFront",
            CdnProvider::Fastly => "Fastly",
            CdnProvider::Cloudflare => "Cloudflare",
            CdnProvider::Akamai => "Akamai",
            CdnProvider::BunnyCdn => "BunnyCDN",
            CdnProvider::Custom(name) => name.as_str(),
        }
    }
}

/// Protocol used to communicate with a CDN origin
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CdnProtocol {
    /// Plain HTTP
    Http,
    /// HTTPS
    Https,
    /// AWS S3 origin
    S3,
    /// RTMP stream origin
    Rtmp,
}

impl CdnProtocol {
    /// Scheme string for the protocol
    #[must_use]
    pub fn scheme(&self) -> &str {
        match self {
            CdnProtocol::Http => "http",
            CdnProtocol::Https => "https",
            CdnProtocol::S3 => "s3",
            CdnProtocol::Rtmp => "rtmp",
        }
    }
}

/// Configuration for a CDN origin server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdnOriginConfig {
    /// Origin URL (hostname or URL)
    pub origin_url: String,
    /// Protocol to use when connecting to the origin
    pub protocol: CdnProtocol,
    /// Connection timeout in seconds
    pub timeout_secs: u32,
}

impl CdnOriginConfig {
    /// Create a new origin configuration
    #[must_use]
    pub fn new(origin_url: impl Into<String>, protocol: CdnProtocol, timeout_secs: u32) -> Self {
        Self {
            origin_url: origin_url.into(),
            protocol,
            timeout_secs,
        }
    }
}

/// An edge location for a CDN PoP (Point of Presence)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdnEdgeLocation {
    /// Geographic region identifier (e.g. "us-east-1")
    pub region: String,
    /// City name
    pub city: String,
    /// Measured latency in milliseconds
    pub latency_ms: u32,
    /// Available egress capacity in Gbps
    pub capacity_gbps: f32,
    /// Approximate latitude of this PoP
    pub latitude: f32,
    /// Approximate longitude of this PoP
    pub longitude: f32,
}

impl CdnEdgeLocation {
    /// Create a new edge location
    #[must_use]
    pub fn new(
        region: impl Into<String>,
        city: impl Into<String>,
        latency_ms: u32,
        capacity_gbps: f32,
        latitude: f32,
        longitude: f32,
    ) -> Self {
        Self {
            region: region.into(),
            city: city.into(),
            latency_ms,
            capacity_gbps,
            latitude,
            longitude,
        }
    }

    /// Haversine distance (km) from a given lat/lon to this edge location
    #[must_use]
    pub fn distance_km(&self, lat: f32, lon: f32) -> f32 {
        const R: f32 = 6371.0; // Earth radius in km
        let dlat = (self.latitude - lat).to_radians();
        let dlon = (self.longitude - lon).to_radians();
        let a = (dlat / 2.0).sin().powi(2)
            + lat.to_radians().cos()
                * self.latitude.to_radians().cos()
                * (dlon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().asin();
        R * c
    }
}

/// A CDN distribution (deployed configuration)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdnDistribution {
    /// Unique distribution identifier
    pub id: String,
    /// Public domain name for this distribution
    pub domain: String,
    /// CDN provider
    pub provider: CdnProvider,
    /// Origin configurations
    pub origins: Vec<CdnOriginConfig>,
    /// Edge locations serving this distribution
    pub edge_locations: Vec<CdnEdgeLocation>,
    /// Whether the distribution is currently enabled
    pub enabled: bool,
}

impl CdnDistribution {
    /// Create a new distribution
    #[must_use]
    pub fn new(id: impl Into<String>, domain: impl Into<String>, provider: CdnProvider) -> Self {
        Self {
            id: id.into(),
            domain: domain.into(),
            provider,
            origins: Vec::new(),
            edge_locations: Vec::new(),
            enabled: true,
        }
    }

    /// Add an origin to the distribution
    pub fn add_origin(&mut self, origin: CdnOriginConfig) {
        self.origins.push(origin);
    }

    /// Add an edge location to the distribution
    pub fn add_edge_location(&mut self, edge: CdnEdgeLocation) {
        self.edge_locations.push(edge);
    }
}

/// Cache policy controlling how and for how long content is cached
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdnCachePolicy {
    /// Time-to-live for cached objects in seconds
    pub ttl_secs: u32,
    /// Stale TTL (serve stale while revalidating) in seconds
    pub stale_ttl_secs: u32,
    /// HTTP headers used to vary the cache key
    pub vary_headers: Vec<String>,
    /// Whether the query string is included in the cache key
    pub cache_query_string: bool,
}

impl CdnCachePolicy {
    /// Create a new cache policy
    #[must_use]
    pub fn new(ttl_secs: u32, stale_ttl_secs: u32, cache_query_string: bool) -> Self {
        Self {
            ttl_secs,
            stale_ttl_secs,
            vary_headers: Vec::new(),
            cache_query_string,
        }
    }

    /// Default policy: 1-hour TTL, no query string caching
    #[must_use]
    pub fn default_policy() -> Self {
        Self::new(3600, 300, false)
    }

    /// Media streaming policy: short TTL, cache query strings (for HLS tokens)
    #[must_use]
    pub fn media_streaming() -> Self {
        Self {
            ttl_secs: 10,
            stale_ttl_secs: 5,
            vary_headers: vec!["Origin".to_string()],
            cache_query_string: true,
        }
    }

    /// Add a Vary header
    pub fn add_vary_header(&mut self, header: impl Into<String>) {
        self.vary_headers.push(header.into());
    }
}

impl Default for CdnCachePolicy {
    fn default() -> Self {
        Self::default_policy()
    }
}

/// Status of a cache invalidation request
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InvalidationStatus {
    /// Submitted but not yet processing
    Pending,
    /// Currently propagating to edge nodes
    InProgress,
    /// Successfully completed
    Completed,
    /// Failed to complete
    Failed,
}

impl InvalidationStatus {
    /// Whether the invalidation has reached a terminal state
    #[must_use]
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            InvalidationStatus::Completed | InvalidationStatus::Failed
        )
    }
}

/// A cache invalidation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdnInvalidation {
    /// Unique invalidation ID
    pub id: String,
    /// Distribution ID this invalidation targets
    pub distribution_id: String,
    /// Path patterns to invalidate (e.g. "/videos/*")
    pub paths: Vec<String>,
    /// Unix timestamp (seconds) when the invalidation was issued
    pub issued_at_secs: u64,
    /// Current status
    pub status: InvalidationStatus,
}

impl CdnInvalidation {
    /// Create a new invalidation record
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        distribution_id: impl Into<String>,
        paths: Vec<String>,
        issued_at_secs: u64,
    ) -> Self {
        Self {
            id: id.into(),
            distribution_id: distribution_id.into(),
            paths,
            issued_at_secs,
            status: InvalidationStatus::Pending,
        }
    }

    /// Mark the invalidation as completed
    pub fn complete(&mut self) {
        self.status = InvalidationStatus::Completed;
    }

    /// Mark the invalidation as failed
    pub fn fail(&mut self) {
        self.status = InvalidationStatus::Failed;
    }
}

/// Aggregated CDN performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CdnMetrics {
    /// Total bytes transferred from edge nodes
    pub bytes_transferred: u64,
    /// Total number of HTTP requests served
    pub requests: u64,
    /// Fraction of requests served from cache (0.0–1.0)
    pub cache_hit_ratio: f32,
    /// Number of requests forwarded to the origin
    pub origin_requests: u64,
}

impl CdnMetrics {
    /// Create new metrics
    #[must_use]
    pub fn new(
        bytes_transferred: u64,
        requests: u64,
        cache_hit_ratio: f32,
        origin_requests: u64,
    ) -> Self {
        Self {
            bytes_transferred,
            requests,
            cache_hit_ratio,
            origin_requests,
        }
    }

    /// Compute cache miss ratio
    #[must_use]
    pub fn cache_miss_ratio(&self) -> f32 {
        1.0 - self.cache_hit_ratio
    }

    /// Bytes served from cache
    #[must_use]
    pub fn bytes_cached(&self) -> u64 {
        (self.bytes_transferred as f64 * self.cache_hit_ratio as f64) as u64
    }
}

/// Manages multiple CDN distributions and their lifecycle
pub struct CdnManager {
    distributions: Vec<CdnDistribution>,
    invalidations: Vec<CdnInvalidation>,
    next_invalidation_seq: u64,
}

impl CdnManager {
    /// Create a new CDN manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            distributions: Vec::new(),
            invalidations: Vec::new(),
            next_invalidation_seq: 1,
        }
    }

    /// Add a distribution to the manager
    pub fn add_distribution(&mut self, dist: CdnDistribution) {
        self.distributions.push(dist);
    }

    /// Remove a distribution by ID
    pub fn remove_distribution(&mut self, dist_id: &str) {
        self.distributions.retain(|d| d.id != dist_id);
    }

    /// Get a distribution by ID
    #[must_use]
    pub fn get_distribution(&self, dist_id: &str) -> Option<&CdnDistribution> {
        self.distributions.iter().find(|d| d.id == dist_id)
    }

    /// List all distributions
    #[must_use]
    pub fn distributions(&self) -> &[CdnDistribution] {
        &self.distributions
    }

    /// Issue a cache invalidation for the given distribution and path patterns.
    ///
    /// Returns the new `CdnInvalidation` record.
    pub fn invalidate(
        &mut self,
        dist_id: &str,
        paths: Vec<String>,
        now_secs: u64,
    ) -> CdnInvalidation {
        let id = format!("inv-{}", self.next_invalidation_seq);
        self.next_invalidation_seq += 1;

        let inv = CdnInvalidation::new(id, dist_id, paths, now_secs);
        self.invalidations.push(inv.clone());
        inv
    }

    /// Return all invalidations for a distribution
    #[must_use]
    pub fn invalidations_for(&self, dist_id: &str) -> Vec<&CdnInvalidation> {
        self.invalidations
            .iter()
            .filter(|i| i.distribution_id == dist_id)
            .collect()
    }

    /// Select the edge location closest to the given geographic coordinates.
    ///
    /// Searches across all distributions.  Returns `None` if no edge locations
    /// are registered.
    #[must_use]
    pub fn get_best_edge(&self, lat: f32, lon: f32) -> Option<&CdnEdgeLocation> {
        self.distributions
            .iter()
            .flat_map(|d| d.edge_locations.iter())
            .min_by(|a, b| {
                a.distance_km(lat, lon)
                    .partial_cmp(&b.distance_km(lat, lon))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

impl Default for CdnManager {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dist(id: &str) -> CdnDistribution {
        CdnDistribution::new(id, format!("{id}.example.com"), CdnProvider::Cloudfront)
    }

    // 1. CdnProvider::name()
    #[test]
    fn test_provider_name() {
        assert_eq!(CdnProvider::Cloudfront.name(), "Amazon CloudFront");
        assert_eq!(CdnProvider::Fastly.name(), "Fastly");
        assert_eq!(CdnProvider::Cloudflare.name(), "Cloudflare");
        assert_eq!(CdnProvider::Akamai.name(), "Akamai");
        assert_eq!(CdnProvider::BunnyCdn.name(), "BunnyCDN");
        assert_eq!(
            CdnProvider::Custom("MyProvider".to_string()).name(),
            "MyProvider"
        );
    }

    // 2. CdnProtocol::scheme()
    #[test]
    fn test_protocol_scheme() {
        assert_eq!(CdnProtocol::Http.scheme(), "http");
        assert_eq!(CdnProtocol::Https.scheme(), "https");
        assert_eq!(CdnProtocol::S3.scheme(), "s3");
        assert_eq!(CdnProtocol::Rtmp.scheme(), "rtmp");
    }

    // 3. CdnOriginConfig construction
    #[test]
    fn test_origin_config_new() {
        let origin = CdnOriginConfig::new("origin.example.com", CdnProtocol::Https, 30);
        assert_eq!(origin.origin_url, "origin.example.com");
        assert_eq!(origin.protocol, CdnProtocol::Https);
        assert_eq!(origin.timeout_secs, 30);
    }

    // 4. CdnDistribution add_origin / add_edge_location
    #[test]
    fn test_distribution_origins_and_edges() {
        let mut dist = make_dist("d-1");
        dist.add_origin(CdnOriginConfig::new(
            "origin.example.com",
            CdnProtocol::Https,
            30,
        ));
        dist.add_edge_location(CdnEdgeLocation::new(
            "us-east-1",
            "New York",
            5,
            100.0,
            40.71,
            -74.01,
        ));

        assert_eq!(dist.origins.len(), 1);
        assert_eq!(dist.edge_locations.len(), 1);
        assert!(dist.enabled);
    }

    // 5. CdnCachePolicy default and media streaming
    #[test]
    fn test_cache_policy_defaults() {
        let default_policy = CdnCachePolicy::default_policy();
        assert_eq!(default_policy.ttl_secs, 3600);
        assert!(!default_policy.cache_query_string);

        let media = CdnCachePolicy::media_streaming();
        assert_eq!(media.ttl_secs, 10);
        assert!(media.cache_query_string);
    }

    // 6. CdnCachePolicy add_vary_header
    #[test]
    fn test_cache_policy_vary_header() {
        let mut policy = CdnCachePolicy::default_policy();
        policy.add_vary_header("Accept-Encoding");
        assert_eq!(policy.vary_headers.len(), 1);
        assert_eq!(policy.vary_headers[0], "Accept-Encoding");
    }

    // 7. InvalidationStatus::is_terminal()
    #[test]
    fn test_invalidation_status_terminal() {
        assert!(!InvalidationStatus::Pending.is_terminal());
        assert!(!InvalidationStatus::InProgress.is_terminal());
        assert!(InvalidationStatus::Completed.is_terminal());
        assert!(InvalidationStatus::Failed.is_terminal());
    }

    // 8. CdnInvalidation complete / fail
    #[test]
    fn test_invalidation_lifecycle() {
        let mut inv =
            CdnInvalidation::new("inv-1", "d-1", vec!["/videos/*".to_string()], 1_700_000_000);
        assert_eq!(inv.status, InvalidationStatus::Pending);
        inv.complete();
        assert_eq!(inv.status, InvalidationStatus::Completed);
        assert!(inv.status.is_terminal());
    }

    // 9. CdnManager add / remove distribution
    #[test]
    fn test_manager_add_remove_distribution() {
        let mut mgr = CdnManager::new();
        mgr.add_distribution(make_dist("d-1"));
        mgr.add_distribution(make_dist("d-2"));
        assert_eq!(mgr.distributions().len(), 2);

        mgr.remove_distribution("d-1");
        assert_eq!(mgr.distributions().len(), 1);
        assert_eq!(mgr.distributions()[0].id, "d-2");
    }

    // 10. CdnManager invalidate
    #[test]
    fn test_manager_invalidate() {
        let mut mgr = CdnManager::new();
        mgr.add_distribution(make_dist("d-1"));

        let inv = mgr.invalidate("d-1", vec!["/*".to_string()], 1_700_000_000);
        assert_eq!(inv.distribution_id, "d-1");
        assert_eq!(inv.paths, vec!["/*"]);
        assert_eq!(inv.status, InvalidationStatus::Pending);

        let pending = mgr.invalidations_for("d-1");
        assert_eq!(pending.len(), 1);
    }

    // 11. CdnManager get_best_edge
    #[test]
    fn test_manager_get_best_edge() {
        let mut mgr = CdnManager::new();
        let mut dist = make_dist("d-1");
        // Tokyo: 35.68N, 139.69E
        dist.add_edge_location(CdnEdgeLocation::new(
            "ap-northeast-1",
            "Tokyo",
            2,
            200.0,
            35.68,
            139.69,
        ));
        // New York: 40.71N, -74.01E
        dist.add_edge_location(CdnEdgeLocation::new(
            "us-east-1",
            "New York",
            120,
            100.0,
            40.71,
            -74.01,
        ));
        mgr.add_distribution(dist);

        // Request from Osaka (34.69N, 135.50E) – closer to Tokyo
        let best = mgr
            .get_best_edge(34.69, 135.50)
            .expect("best should be valid");
        assert_eq!(best.city, "Tokyo");
    }

    // 12. CdnMetrics computed values
    #[test]
    fn test_cdn_metrics() {
        let metrics = CdnMetrics::new(1_000_000, 10_000, 0.85, 1500);
        assert!((metrics.cache_miss_ratio() - 0.15).abs() < 1e-5);
        assert_eq!(metrics.bytes_cached(), 850_000);
    }

    // 13. Sequential invalidation IDs
    #[test]
    fn test_sequential_invalidation_ids() {
        let mut mgr = CdnManager::new();
        mgr.add_distribution(make_dist("d-1"));
        let inv1 = mgr.invalidate("d-1", vec![], 0);
        let inv2 = mgr.invalidate("d-1", vec![], 0);
        assert_ne!(inv1.id, inv2.id);
    }

    // 14. No distributions – get_best_edge returns None
    #[test]
    fn test_get_best_edge_empty() {
        let mgr = CdnManager::new();
        assert!(mgr.get_best_edge(0.0, 0.0).is_none());
    }
}
