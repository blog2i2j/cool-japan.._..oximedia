//! CDN request routing — select the best PoP for a client request based on
//! geography, load, and health.
//!
//! # Strategies
//!
//! | Variant | Description |
//! |---------|-------------|
//! | [`RoutingStrategy::LatencyBased`] | Prefer the PoP with lowest estimated propagation latency (Haversine if lat/lon available, else region match). |
//! | [`RoutingStrategy::LoadBased`] | Prefer the healthy PoP with the lowest `load_pct`. |
//! | [`RoutingStrategy::GeoNearest`] | Prefer the geographically nearest healthy PoP by Haversine distance. |
//! | [`RoutingStrategy::RoundRobin`] | Cycle through healthy PoPs in insertion order. |
//! | [`RoutingStrategy::WeightedRandom`] | Sample proportional to `(1 − load_pct) × capacity_gbps`; falls back to uniform when all weights are zero. |
//!
//! # Notes
//!
//! - **No I/O** is performed; the router is a pure-logic component.
//! - Unhealthy PoPs (`healthy == false`) are excluded from all strategies.
//! - When no healthy PoPs exist, [`RequestRouter::route`] returns `None`.

use std::sync::atomic::{AtomicUsize, Ordering};

// ─── Haversine (local copy to avoid a hard dep on geo_routing) ────────────────

fn haversine_km(lat1: f32, lon1: f32, lat2: f32, lon2: f32) -> f32 {
    const R_KM: f32 = 6_371.0;
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let lat1_r = lat1.to_radians();
    let lat2_r = lat2.to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1_r.cos() * lat2_r.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();
    R_KM * c
}

// ─── CdnPop ───────────────────────────────────────────────────────────────────

/// A CDN Point-of-Presence (PoP) that can serve client requests.
#[derive(Debug, Clone)]
pub struct CdnPop {
    /// Unique identifier (e.g. `"iad01"`, `"lhr05"`).
    pub id: String,
    /// Broad region label (e.g. `"us-east-1"`, `"eu-west-1"`).
    pub region: String,
    /// ISO 3166-1 alpha-2 country code (e.g. `"US"`, `"GB"`).
    pub country: String,
    /// Latitude in decimal degrees.
    pub lat: f32,
    /// Longitude in decimal degrees.
    pub lon: f32,
    /// Current load as a fraction in [0, 1].  0 = idle, 1 = fully saturated.
    pub load_pct: f32,
    /// Whether this PoP is accepting traffic.
    pub healthy: bool,
    /// Installed capacity in Gbit/s.
    pub capacity_gbps: f32,
}

impl CdnPop {
    /// Construct a new PoP entry.
    pub fn new(
        id: impl Into<String>,
        region: impl Into<String>,
        country: impl Into<String>,
        lat: f32,
        lon: f32,
        load_pct: f32,
        healthy: bool,
        capacity_gbps: f32,
    ) -> Self {
        Self {
            id: id.into(),
            region: region.into(),
            country: country.into(),
            lat,
            lon,
            load_pct: load_pct.clamp(0.0, 1.0),
            healthy,
            capacity_gbps: capacity_gbps.max(0.0),
        }
    }

    /// Compute the Haversine distance in km to a client position.
    pub fn distance_to(&self, lat: f32, lon: f32) -> f32 {
        haversine_km(self.lat, self.lon, lat, lon)
    }

    /// Weighted-random selection weight: `(1 − load) × capacity`.
    ///
    /// Returns 0 when the PoP is fully loaded or has no capacity.
    pub fn selection_weight(&self) -> f32 {
        (1.0 - self.load_pct).max(0.0) * self.capacity_gbps
    }
}

// ─── RoutingStrategy ─────────────────────────────────────────────────────────

/// Algorithm used to pick a target PoP for an incoming request.
#[derive(Debug, Clone, PartialEq)]
pub enum RoutingStrategy {
    /// Lowest propagation latency (distance-based or region-based fallback).
    LatencyBased,
    /// Healthy PoP with the smallest `load_pct`.
    LoadBased,
    /// Geographically nearest healthy PoP (requires client coordinates).
    GeoNearest,
    /// Round-robin over healthy PoPs in registration order.
    RoundRobin,
    /// Probabilistic selection weighted by `(1 − load) × capacity_gbps`.
    WeightedRandom,
}

// ─── RouteRequest ────────────────────────────────────────────────────────────

/// Metadata about an incoming client request used by the router.
#[derive(Debug, Clone)]
pub struct RouteRequest {
    /// Broad region of the client (e.g. `"us-east-1"`).  Used as a fallback
    /// for strategies that need geographic information but receive no lat/lon.
    pub client_ip_region: String,
    /// Client latitude, if available from GeoIP enrichment.
    pub client_lat: Option<f32>,
    /// Client longitude, if available from GeoIP enrichment.
    pub client_lon: Option<f32>,
    /// Size of the requested asset in bytes (informational; may influence
    /// capacity checks in the future).
    pub asset_size_bytes: u64,
}

impl RouteRequest {
    /// Create a request with explicit coordinates.
    pub fn with_coords(
        region: impl Into<String>,
        lat: f32,
        lon: f32,
        asset_size_bytes: u64,
    ) -> Self {
        Self {
            client_ip_region: region.into(),
            client_lat: Some(lat),
            client_lon: Some(lon),
            asset_size_bytes,
        }
    }

    /// Create a request with only a region label (no coordinates).
    pub fn region_only(region: impl Into<String>, asset_size_bytes: u64) -> Self {
        Self {
            client_ip_region: region.into(),
            client_lat: None,
            client_lon: None,
            asset_size_bytes,
        }
    }
}

// ─── RequestRouter ───────────────────────────────────────────────────────────

/// Routes incoming CDN requests to the most suitable PoP.
///
/// The router holds an ordered list of registered PoPs and a per-instance
/// round-robin counter.
pub struct RequestRouter {
    pops: Vec<CdnPop>,
    rr_counter: AtomicUsize,
}

impl std::fmt::Debug for RequestRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RequestRouter")
            .field("pops", &self.pops)
            .finish_non_exhaustive()
    }
}

impl Default for RequestRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestRouter {
    /// Create an empty router.
    pub fn new() -> Self {
        Self {
            pops: Vec::new(),
            rr_counter: AtomicUsize::new(0),
        }
    }

    /// Register a PoP.
    pub fn add_pop(&mut self, pop: CdnPop) {
        self.pops.push(pop);
    }

    /// Remove a PoP by ID.  Returns `true` if a PoP was found and removed.
    pub fn remove_pop(&mut self, id: &str) -> bool {
        let before = self.pops.len();
        self.pops.retain(|p| p.id != id);
        self.pops.len() < before
    }

    /// Return references to all healthy PoPs.
    pub fn healthy_pops(&self) -> Vec<&CdnPop> {
        self.pops.iter().filter(|p| p.healthy).collect()
    }

    /// Return the total number of registered PoPs (healthy or not).
    pub fn pop_count(&self) -> usize {
        self.pops.len()
    }

    /// Select the best PoP for `request` using `strategy`.
    ///
    /// Returns `None` when no healthy PoPs exist.
    pub fn route(
        &self,
        request: &RouteRequest,
        strategy: RoutingStrategy,
    ) -> Option<&CdnPop> {
        match strategy {
            RoutingStrategy::LatencyBased => self.route_latency(request),
            RoutingStrategy::LoadBased => self.route_load(),
            RoutingStrategy::GeoNearest => self.route_geo_nearest(request),
            RoutingStrategy::RoundRobin => self.route_round_robin(),
            RoutingStrategy::WeightedRandom => self.route_weighted_random(),
        }
    }

    // ── Strategy implementations ─────────────────────────────────────────

    /// Latency-based: use Haversine if coordinates are available, otherwise
    /// prefer PoPs whose region string contains the client region as a prefix.
    fn route_latency(&self, request: &RouteRequest) -> Option<&CdnPop> {
        if let (Some(clat), Some(clon)) = (request.client_lat, request.client_lon) {
            // Prefer closest by propagation distance
            self.pops
                .iter()
                .filter(|p| p.healthy)
                .min_by(|a, b| {
                    let da = a.distance_to(clat, clon);
                    let db = b.distance_to(clat, clon);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
        } else {
            // Fallback: region-string prefix match, then load as tiebreak
            let region = &request.client_ip_region;
            // Prefer an exact region match first
            let exact = self
                .pops
                .iter()
                .filter(|p| p.healthy && &p.region == region)
                .min_by(|a, b| {
                    a.load_pct
                        .partial_cmp(&b.load_pct)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            if exact.is_some() {
                return exact;
            }
            // Fallback: prefix match (e.g. `us-east-1` matches `us-east-2`)
            let prefix_len = region.find('-').map(|i| i + 1).unwrap_or(region.len());
            let prefix = &region[..prefix_len];
            let prefix_match = self
                .pops
                .iter()
                .filter(|p| p.healthy && p.region.starts_with(prefix))
                .min_by(|a, b| {
                    a.load_pct
                        .partial_cmp(&b.load_pct)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            if prefix_match.is_some() {
                return prefix_match;
            }
            // Ultimate fallback: any healthy PoP with lowest load
            self.route_load()
        }
    }

    /// Load-based: healthy PoP with the lowest `load_pct`.
    fn route_load(&self) -> Option<&CdnPop> {
        self.pops
            .iter()
            .filter(|p| p.healthy)
            .min_by(|a, b| {
                a.load_pct
                    .partial_cmp(&b.load_pct)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Geo-nearest: healthy PoP closest by Haversine distance.
    /// Falls back to load-based when the request carries no coordinates.
    fn route_geo_nearest(&self, request: &RouteRequest) -> Option<&CdnPop> {
        match (request.client_lat, request.client_lon) {
            (Some(clat), Some(clon)) => self
                .pops
                .iter()
                .filter(|p| p.healthy)
                .min_by(|a, b| {
                    let da = a.distance_to(clat, clon);
                    let db = b.distance_to(clat, clon);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                }),
            _ => self.route_load(),
        }
    }

    /// Round-robin over healthy PoPs (wraps using the atomic counter).
    fn route_round_robin(&self) -> Option<&CdnPop> {
        let healthy: Vec<&CdnPop> = self.pops.iter().filter(|p| p.healthy).collect();
        if healthy.is_empty() {
            return None;
        }
        let idx = self
            .rr_counter
            .fetch_add(1, Ordering::Relaxed)
            % healthy.len();
        Some(healthy[idx])
    }

    /// Weighted-random selection: weight = `(1 − load_pct) × capacity_gbps`.
    ///
    /// Uses a deterministic pseudo-random algorithm (xorshift64 seeded by the
    /// counter) to avoid requiring the `rand` crate.
    fn route_weighted_random(&self) -> Option<&CdnPop> {
        let healthy: Vec<&CdnPop> = self.pops.iter().filter(|p| p.healthy).collect();
        if healthy.is_empty() {
            return None;
        }

        let weights: Vec<f32> = healthy.iter().map(|p| p.selection_weight()).collect();
        let total: f32 = weights.iter().sum();

        if total <= 0.0 {
            // All weights are zero → fall back to uniform round-robin
            let idx = self
                .rr_counter
                .fetch_add(1, Ordering::Relaxed)
                % healthy.len();
            return Some(healthy[idx]);
        }

        // xorshift64 seeded by counter to get a pseudo-random pick
        let seed = self.rr_counter.fetch_add(1, Ordering::Relaxed) as u64 + 1;
        let rand_val = xorshift64(seed);
        // Map to [0, total)
        let threshold = (rand_val as f64 / u64::MAX as f64) as f32 * total;

        let mut cumulative = 0.0_f32;
        for (pop, &w) in healthy.iter().zip(weights.iter()) {
            cumulative += w;
            if cumulative >= threshold {
                return Some(pop);
            }
        }
        // Rounding fallback: last element
        healthy.last().copied()
    }
}

/// Xorshift64 PRNG — pure-Rust, no `rand` dependency.
fn xorshift64(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn pop(id: &str, region: &str, lat: f32, lon: f32, load: f32, cap: f32) -> CdnPop {
        CdnPop::new(id, region, "US", lat, lon, load, true, cap)
    }

    fn sick_pop(id: &str, region: &str) -> CdnPop {
        CdnPop::new(id, region, "US", 0.0, 0.0, 0.5, false, 10.0)
    }

    // 1. empty router returns None
    #[test]
    fn test_empty_returns_none() {
        let router = RequestRouter::new();
        let req = RouteRequest::region_only("us-east-1", 1_000);
        assert!(router.route(&req, RoutingStrategy::LoadBased).is_none());
        assert!(router
            .route(&req, RoutingStrategy::RoundRobin)
            .is_none());
        assert!(router
            .route(&req, RoutingStrategy::WeightedRandom)
            .is_none());
    }

    // 2. load-based selects the least-loaded healthy PoP
    #[test]
    fn test_load_based_selects_least_loaded() {
        let mut r = RequestRouter::new();
        r.add_pop(pop("high", "us-east-1", 40.0, -74.0, 0.9, 10.0));
        r.add_pop(pop("low", "us-east-1", 40.0, -74.0, 0.1, 10.0));
        r.add_pop(pop("mid", "us-east-1", 40.0, -74.0, 0.5, 10.0));
        let req = RouteRequest::region_only("us-east-1", 1_000);
        let selected = r.route(&req, RoutingStrategy::LoadBased).expect("some");
        assert_eq!(selected.id, "low");
    }

    // 3. unhealthy PoPs are excluded from load-based
    #[test]
    fn test_load_based_excludes_unhealthy() {
        let mut r = RequestRouter::new();
        r.add_pop(sick_pop("bad", "us-east-1"));
        r.add_pop(pop("good", "us-east-1", 40.0, -74.0, 0.8, 10.0));
        let req = RouteRequest::region_only("us-east-1", 1_000);
        let selected = r.route(&req, RoutingStrategy::LoadBased).expect("some");
        assert_eq!(selected.id, "good");
    }

    // 4. geo-nearest selects correct PoP by coordinates
    #[test]
    fn test_geo_nearest_correct() {
        let mut r = RequestRouter::new();
        // London (51.5, -0.1) and New York (40.7, -74.0)
        // Client in Paris (48.8, 2.3) → London closer
        r.add_pop(pop("london", "eu-west-1", 51.5, -0.1, 0.3, 10.0));
        r.add_pop(pop("new-york", "us-east-1", 40.7, -74.0, 0.3, 10.0));
        let req = RouteRequest::with_coords("eu-west-1", 48.8, 2.3, 1_000);
        let selected = r.route(&req, RoutingStrategy::GeoNearest).expect("some");
        assert_eq!(selected.id, "london");
    }

    // 5. geo-nearest excludes unhealthy PoPs
    #[test]
    fn test_geo_nearest_excludes_unhealthy() {
        let mut r = RequestRouter::new();
        // Near PoP is unhealthy; far PoP is healthy
        let mut near = pop("near", "eu-west-1", 48.9, 2.4, 0.1, 10.0);
        near.healthy = false;
        r.add_pop(near);
        r.add_pop(pop("far", "us-east-1", 40.7, -74.0, 0.3, 10.0));
        let req = RouteRequest::with_coords("eu-west-1", 48.8, 2.3, 1_000);
        let selected = r.route(&req, RoutingStrategy::GeoNearest).expect("some");
        assert_eq!(selected.id, "far");
    }

    // 6. round-robin cycles through healthy PoPs
    #[test]
    fn test_round_robin_cycles() {
        let mut r = RequestRouter::new();
        r.add_pop(pop("a", "us-east-1", 40.0, -74.0, 0.3, 10.0));
        r.add_pop(pop("b", "us-east-1", 40.0, -74.0, 0.3, 10.0));
        r.add_pop(pop("c", "us-east-1", 40.0, -74.0, 0.3, 10.0));

        let req = RouteRequest::region_only("us-east-1", 1_000);
        let ids: Vec<String> = (0..6)
            .map(|_| {
                r.route(&req, RoutingStrategy::RoundRobin)
                    .expect("some")
                    .id
                    .clone()
            })
            .collect();
        // The pattern should repeat after 3 requests
        assert_eq!(ids[0], ids[3]);
        assert_eq!(ids[1], ids[4]);
        assert_eq!(ids[2], ids[5]);
    }

    // 7. round-robin skips unhealthy PoPs
    #[test]
    fn test_round_robin_skips_unhealthy() {
        let mut r = RequestRouter::new();
        r.add_pop(sick_pop("bad", "us-east-1"));
        r.add_pop(pop("good", "us-east-1", 40.0, -74.0, 0.3, 10.0));
        let req = RouteRequest::region_only("us-east-1", 1_000);
        for _ in 0..5 {
            let sel = r.route(&req, RoutingStrategy::RoundRobin).expect("some");
            assert_eq!(sel.id, "good");
        }
    }

    // 8. weighted-random returns a healthy PoP
    #[test]
    fn test_weighted_random_returns_healthy() {
        let mut r = RequestRouter::new();
        r.add_pop(sick_pop("bad", "us-east-1"));
        r.add_pop(pop("good1", "us-east-1", 40.0, -74.0, 0.2, 10.0));
        r.add_pop(pop("good2", "us-east-1", 40.0, -74.0, 0.5, 20.0));
        let req = RouteRequest::region_only("us-east-1", 1_000);
        for _ in 0..20 {
            let sel = r
                .route(&req, RoutingStrategy::WeightedRandom)
                .expect("some");
            assert!(sel.healthy, "selected PoP must be healthy");
        }
    }

    // 9. latency-based with coordinates picks the nearest PoP
    #[test]
    fn test_latency_based_with_coords() {
        let mut r = RequestRouter::new();
        r.add_pop(pop("close", "eu-west-1", 51.5, -0.1, 0.5, 10.0));
        r.add_pop(pop("far", "us-east-1", 40.7, -74.0, 0.1, 10.0));
        // Client near London
        let req = RouteRequest::with_coords("eu-west-1", 51.0, -0.2, 1_000);
        let selected = r.route(&req, RoutingStrategy::LatencyBased).expect("some");
        assert_eq!(selected.id, "close");
    }

    // 10. latency-based without coords falls back to region match
    #[test]
    fn test_latency_based_region_fallback() {
        let mut r = RequestRouter::new();
        r.add_pop(pop("eu-pop", "eu-west-1", 51.5, -0.1, 0.7, 10.0));
        r.add_pop(pop("us-pop", "us-east-1", 40.7, -74.0, 0.2, 10.0));
        // Client region us-east-1 but no coordinates
        let req = RouteRequest::region_only("us-east-1", 1_000);
        let selected = r.route(&req, RoutingStrategy::LatencyBased).expect("some");
        assert_eq!(selected.id, "us-pop");
    }

    // 11. remove_pop removes a PoP
    #[test]
    fn test_remove_pop() {
        let mut r = RequestRouter::new();
        r.add_pop(pop("a", "us-east-1", 40.0, -74.0, 0.3, 10.0));
        r.add_pop(pop("b", "us-east-1", 40.0, -74.0, 0.3, 10.0));
        assert!(r.remove_pop("a"));
        assert_eq!(r.pop_count(), 1);
        assert!(!r.remove_pop("a")); // already gone
    }

    // 12. healthy_pops only returns healthy ones
    #[test]
    fn test_healthy_pops_filter() {
        let mut r = RequestRouter::new();
        r.add_pop(pop("ok", "us-east-1", 40.0, -74.0, 0.3, 10.0));
        r.add_pop(sick_pop("bad", "us-east-1"));
        assert_eq!(r.healthy_pops().len(), 1);
        assert_eq!(r.healthy_pops()[0].id, "ok");
    }

    // 13. weighted-random with all-zero weights falls back gracefully
    #[test]
    fn test_weighted_random_all_zero_weights() {
        let mut r = RequestRouter::new();
        // Fully-loaded PoP has weight 0
        r.add_pop(CdnPop::new(
            "full",
            "us-east-1",
            "US",
            40.0,
            -74.0,
            1.0, // load_pct = 1.0 → weight = 0
            true,
            10.0,
        ));
        let req = RouteRequest::region_only("us-east-1", 1_000);
        let sel = r.route(&req, RoutingStrategy::WeightedRandom).expect("some");
        assert_eq!(sel.id, "full");
    }

    // 14. selection_weight calculation
    #[test]
    fn test_selection_weight() {
        let p = pop("x", "us", 0.0, 0.0, 0.25, 40.0);
        // (1 - 0.25) * 40 = 30
        let w = p.selection_weight();
        assert!((w - 30.0).abs() < 1e-4, "w={w}");
    }

    // 15. all-unhealthy returns None for every strategy
    #[test]
    fn test_all_unhealthy_returns_none() {
        let mut r = RequestRouter::new();
        r.add_pop(sick_pop("bad1", "us-east-1"));
        r.add_pop(sick_pop("bad2", "eu-west-1"));
        let req = RouteRequest::with_coords("us-east-1", 40.0, -74.0, 1_000);
        for strat in [
            RoutingStrategy::LoadBased,
            RoutingStrategy::GeoNearest,
            RoutingStrategy::LatencyBased,
            RoutingStrategy::RoundRobin,
            RoutingStrategy::WeightedRandom,
        ] {
            assert!(
                r.route(&req, strat).is_none(),
                "all-unhealthy must return None"
            );
        }
    }
}
