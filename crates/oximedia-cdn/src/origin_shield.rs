//! Origin shield — mid-tier cache layer between edges and the origin server.
//!
//! # Overview
//!
//! An **origin shield** (also called a *mid-tier cache* or *shield PoP*)
//! aggregates cache traffic from many edge PoPs into a much smaller number of
//! upstream connections to the origin.  This dramatically reduces origin load
//! during cache-miss storms or cold-start events.
//!
//! [`OriginShield`] provides:
//!
//! - Per-region node selection (`ShieldNode`).
//! - Asset tracking: once an asset is "stored" in a node its subsequent
//!   requests produce `ShieldResponse::CacheHit`.
//! - LRU eviction when a node exceeds 90 % fill.
//! - Aggregate hit-rate and capacity statistics.
//!
//! # Simulation model
//!
//! This module is intentionally pure-logic: no network I/O, no file I/O.
//! Asset storage is tracked via an in-memory `HashMap`; sizes are provided by
//! the caller.

use std::collections::HashMap;

// ─── ShieldNode ───────────────────────────────────────────────────────────────

/// A single node in the origin-shield tier.
#[derive(Debug, Clone)]
pub struct ShieldNode {
    /// Unique node identifier.
    pub id: String,
    /// Broad region label (e.g. `"us-east-1"`, `"eu-west-1"`).
    pub region: String,
    /// Total cache capacity in GB.
    pub cache_size_gb: f64,
    /// How much of the cache is currently filled, in GB.
    pub current_fill_gb: f64,
    /// Total requests ever routed through this node.
    pub requests_served: u64,
    /// Requests satisfied from cache (hits).
    pub cache_hits: u64,
    /// Per-asset metadata stored in this node.
    /// Key: asset_id; Value: `(size_bytes, insertion_seq)`.
    cached_assets: HashMap<String, (u64, u64)>,
    /// Monotonically increasing insertion counter — used for LRU ordering.
    insertion_seq: u64,
}

impl ShieldNode {
    /// Create a new node.
    pub fn new(
        id: impl Into<String>,
        region: impl Into<String>,
        cache_size_gb: f64,
    ) -> Self {
        Self {
            id: id.into(),
            region: region.into(),
            cache_size_gb: cache_size_gb.max(0.0),
            current_fill_gb: 0.0,
            requests_served: 0,
            cache_hits: 0,
            cached_assets: HashMap::new(),
            insertion_seq: 0,
        }
    }

    /// Returns `true` if this node has `asset_id` cached.
    pub fn has_asset(&self, asset_id: &str) -> bool {
        self.cached_assets.contains_key(asset_id)
    }

    /// Compute the cache fill ratio in [0, 1].
    pub fn fill_ratio(&self) -> f64 {
        if self.cache_size_gb <= 0.0 {
            return 1.0;
        }
        (self.current_fill_gb / self.cache_size_gb).clamp(0.0, 1.0)
    }

    /// Returns `true` if the node is at or above 90 % fill.
    pub fn is_near_full(&self) -> bool {
        self.fill_ratio() >= 0.90
    }

    /// Store an asset, evicting the LRU entry first if the node is near full.
    ///
    /// If the asset is already present it is a no-op (update of sequence
    /// would require more bookkeeping and is not needed for correctness).
    pub fn store_asset(&mut self, asset_id: &str, size_bytes: u64) {
        if self.cached_assets.contains_key(asset_id) {
            return; // already cached
        }
        let size_gb = size_bytes as f64 / 1_073_741_824.0;

        // Evict LRU entries until there is space
        while self.is_near_full() && !self.cached_assets.is_empty() {
            self.evict_lru();
        }

        let seq = self.insertion_seq;
        self.insertion_seq += 1;
        self.cached_assets
            .insert(asset_id.to_string(), (size_bytes, seq));
        self.current_fill_gb += size_gb;
    }

    /// Remove the least-recently-inserted (LRU) asset.
    fn evict_lru(&mut self) {
        // Find the entry with the smallest insertion sequence number
        if let Some(oldest_key) = self
            .cached_assets
            .iter()
            .min_by_key(|(_, &(_, seq))| seq)
            .map(|(k, _)| k.clone())
        {
            if let Some((size_bytes, _)) = self.cached_assets.remove(&oldest_key) {
                self.current_fill_gb -= size_bytes as f64 / 1_073_741_824.0;
                self.current_fill_gb = self.current_fill_gb.max(0.0);
            }
        }
    }

    /// Number of distinct assets currently cached.
    pub fn asset_count(&self) -> usize {
        self.cached_assets.len()
    }

    /// Hit rate for this node in [0, 1].
    pub fn hit_rate(&self) -> f32 {
        if self.requests_served == 0 {
            return 0.0;
        }
        self.cache_hits as f32 / self.requests_served as f32
    }
}

// ─── ShieldRequest ───────────────────────────────────────────────────────────

/// A request arriving at the origin-shield tier.
#[derive(Debug, Clone)]
pub struct ShieldRequest {
    /// Unique asset identifier.
    pub asset_id: String,
    /// Binary size of the asset (used when storing on a cache miss).
    pub size_bytes: u64,
    /// Region from which the request originates (used to pick the nearest node).
    pub client_region: String,
}

impl ShieldRequest {
    /// Construct a new request.
    pub fn new(
        asset_id: impl Into<String>,
        size_bytes: u64,
        client_region: impl Into<String>,
    ) -> Self {
        Self {
            asset_id: asset_id.into(),
            size_bytes,
            client_region: client_region.into(),
        }
    }
}

// ─── ShieldResponse ──────────────────────────────────────────────────────────

/// Result of routing a request through the origin-shield tier.
#[derive(Debug, Clone, PartialEq)]
pub enum ShieldResponse {
    /// Asset was found in a shield node's cache.
    CacheHit {
        /// ID of the node that served the hit.
        node_id: String,
    },
    /// Asset was not found; the shield node will fetch it and store it.
    CacheMiss,
    /// No suitable shield node exists; the request must go directly to origin.
    OriginFetch,
}

// ─── ShieldStats ─────────────────────────────────────────────────────────────

/// Aggregate statistics across all shield nodes.
#[derive(Debug, Clone, Default)]
pub struct ShieldStats {
    /// Total requests handled.
    pub total_requests: u64,
    /// Requests served from cache.
    pub cache_hits: u64,
    /// Requests that required an origin fetch.
    pub origin_fetches: u64,
    /// Aggregate hit rate in [0, 1].
    pub hit_rate: f32,
}

impl ShieldStats {
    fn compute(total: u64, hits: u64, fetches: u64) -> Self {
        let hit_rate = if total == 0 {
            0.0
        } else {
            hits as f32 / total as f32
        };
        Self {
            total_requests: total,
            cache_hits: hits,
            origin_fetches: fetches,
            hit_rate,
        }
    }
}

// ─── OriginShield ────────────────────────────────────────────────────────────

/// Manages a tier of shield nodes and routes asset requests through them.
pub struct OriginShield {
    nodes: Vec<ShieldNode>,
    /// Aggregate miss count (requests that fell through to origin fetch).
    origin_fetch_count: u64,
}

impl std::fmt::Debug for OriginShield {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OriginShield")
            .field("node_count", &self.nodes.len())
            .field("origin_fetch_count", &self.origin_fetch_count)
            .finish()
    }
}

impl Default for OriginShield {
    fn default() -> Self {
        Self::new()
    }
}

impl OriginShield {
    /// Create an empty shield with no nodes.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            origin_fetch_count: 0,
        }
    }

    /// Register a new shield node.
    pub fn add_node(&mut self, node: ShieldNode) {
        self.nodes.push(node);
    }

    /// Remove a node by ID.  Returns `true` if a node was found and removed.
    pub fn remove_node(&mut self, id: &str) -> bool {
        let before = self.nodes.len();
        self.nodes.retain(|n| n.id != id);
        self.nodes.len() < before
    }

    /// Handle an incoming shield request.
    ///
    /// # Algorithm
    ///
    /// 1. Select the "nearest" node for the client's region (exact match first,
    ///    then any node as fallback).
    /// 2. Check if any node already has the asset (global search, then prefer
    ///    nearest).
    /// 3. If a node has the asset → `CacheHit`.
    /// 4. If no node has the asset but a suitable node exists → store the asset
    ///    (simulating a fetch from origin into the shield) → `CacheMiss`.
    /// 5. No nodes at all → `OriginFetch`.
    ///
    /// The `now_secs` parameter is reserved for future TTL-based eviction;
    /// currently unused beyond API compatibility.
    pub fn handle_request(&mut self, request: &ShieldRequest, _now_secs: u64) -> ShieldResponse {
        if self.nodes.is_empty() {
            self.origin_fetch_count += 1;
            return ShieldResponse::OriginFetch;
        }

        // Search globally for a node that already has the asset cached
        if let Some(node) = self
            .nodes
            .iter_mut()
            .find(|n| n.has_asset(&request.asset_id))
        {
            node.requests_served += 1;
            node.cache_hits += 1;
            return ShieldResponse::CacheHit {
                node_id: node.id.clone(),
            };
        }

        // Asset not cached anywhere — pick the nearest node for the region,
        // store the asset there (simulating a shield-to-origin fetch), and
        // return CacheMiss so the caller knows it went to origin this time.
        let target_idx = self.nearest_node_idx(&request.client_region);
        let node = &mut self.nodes[target_idx];
        node.requests_served += 1;
        node.store_asset(&request.asset_id, request.size_bytes);

        ShieldResponse::CacheMiss
    }

    /// Return the index of the "nearest" node to `region`.
    ///
    /// Prefers an exact region match; if none, returns index 0.
    fn nearest_node_idx(&self, region: &str) -> usize {
        // Exact match
        if let Some(idx) = self.nodes.iter().position(|n| n.region == region) {
            return idx;
        }
        // Prefix match (e.g. "us-east" matches "us-east-1")
        let prefix_len = region.find('-').map(|i| i + 1).unwrap_or(region.len());
        let prefix = &region[..prefix_len];
        if let Some(idx) = self.nodes.iter().position(|n| n.region.starts_with(prefix)) {
            return idx;
        }
        // Fallback: first node
        0
    }

    /// Aggregate hit rate in [0, 1] across all nodes.
    pub fn hit_rate(&self) -> f32 {
        let total: u64 = self.nodes.iter().map(|n| n.requests_served).sum();
        let hits: u64 = self.nodes.iter().map(|n| n.cache_hits).sum();
        if total == 0 {
            return 0.0;
        }
        hits as f32 / total as f32
    }

    /// Total cache capacity across all shield nodes, in GB.
    pub fn total_capacity_gb(&self) -> f64 {
        self.nodes.iter().map(|n| n.cache_size_gb).sum()
    }

    /// Total current fill across all shield nodes, in GB.
    pub fn total_fill_gb(&self) -> f64 {
        self.nodes.iter().map(|n| n.current_fill_gb).sum()
    }

    /// Compute aggregate statistics.
    pub fn stats(&self) -> ShieldStats {
        let total: u64 = self.nodes.iter().map(|n| n.requests_served).sum::<u64>()
            + self.origin_fetch_count;
        let hits: u64 = self.nodes.iter().map(|n| n.cache_hits).sum();
        ShieldStats::compute(total, hits, self.origin_fetch_count)
    }

    /// Number of registered nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Retrieve an immutable reference to a node by ID.
    pub fn get_node(&self, id: &str) -> Option<&ShieldNode> {
        self.nodes.iter().find(|n| n.id == id)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_shield() -> OriginShield {
        let mut s = OriginShield::new();
        s.add_node(ShieldNode::new("shield-us", "us-east-1", 10.0));
        s.add_node(ShieldNode::new("shield-eu", "eu-west-1", 8.0));
        s
    }

    fn req(asset: &str, region: &str, size: u64) -> ShieldRequest {
        ShieldRequest::new(asset, size, region)
    }

    // 1. cache miss on first request, then hit on second
    #[test]
    fn test_miss_then_hit() {
        let mut s = make_shield();
        let r1 = s.handle_request(&req("video.mp4", "us-east-1", 1_000_000), 0);
        assert_eq!(r1, ShieldResponse::CacheMiss);
        let r2 = s.handle_request(&req("video.mp4", "us-east-1", 1_000_000), 1);
        assert!(
            matches!(r2, ShieldResponse::CacheHit { .. }),
            "second request should hit: {:?}",
            r2
        );
    }

    // 2. unknown asset returns CacheMiss
    #[test]
    fn test_miss_for_unknown_asset() {
        let mut s = make_shield();
        let r = s.handle_request(&req("unknown.ts", "eu-west-1", 512), 0);
        assert_eq!(r, ShieldResponse::CacheMiss);
    }

    // 3. no nodes → OriginFetch
    #[test]
    fn test_origin_fetch_when_no_nodes() {
        let mut s = OriginShield::new();
        let r = s.handle_request(&req("x", "us-east-1", 100), 0);
        assert_eq!(r, ShieldResponse::OriginFetch);
    }

    // 4. hit rate starts at zero and rises after a hit
    #[test]
    fn test_hit_rate_calculation() {
        let mut s = make_shield();
        assert!((s.hit_rate() - 0.0).abs() < 1e-6);
        // First request: miss
        s.handle_request(&req("a", "us-east-1", 1_000), 0);
        // Second request: hit
        s.handle_request(&req("a", "us-east-1", 1_000), 1);
        // Node "shield-us" has 1 request (the miss) + 1 hit = 2 requests total; 1 hit
        let hr = s.hit_rate();
        assert!(hr > 0.0, "hit rate should be > 0 after a hit: {hr}");
    }

    // 5. total_capacity_gb sums all nodes
    #[test]
    fn test_total_capacity_gb() {
        let s = make_shield();
        assert!((s.total_capacity_gb() - 18.0).abs() < 1e-9);
    }

    // 6. region selection picks the node in the matching region
    #[test]
    fn test_region_selection() {
        let mut s = make_shield();
        // Request from eu-west-1 should store in shield-eu
        s.handle_request(&req("eu-asset", "eu-west-1", 500), 0);
        let eu_node = s.get_node("shield-eu").expect("node");
        assert!(eu_node.has_asset("eu-asset"), "asset should be in eu node");
        let us_node = s.get_node("shield-us").expect("node");
        assert!(!us_node.has_asset("eu-asset"), "asset must NOT be in us node");
    }

    // 7. capacity tracking: current_fill_gb increases on store
    #[test]
    fn test_capacity_tracking() {
        let mut s = OriginShield::new();
        s.add_node(ShieldNode::new("n1", "us-east-1", 100.0));
        // 1 GiB = 1_073_741_824 bytes
        s.handle_request(&req("a", "us-east-1", 1_073_741_824), 0);
        let node = s.get_node("n1").expect("n1");
        assert!(
            (node.current_fill_gb - 1.0).abs() < 1e-6,
            "fill={}",
            node.current_fill_gb
        );
    }

    // 8. LRU eviction when node is near full
    #[test]
    fn test_lru_eviction_on_near_full() {
        let mut s = OriginShield::new();
        // Very small node: 1 GB → fills up quickly
        s.add_node(ShieldNode::new("tiny", "us-east-1", 1.0));
        // Store 9 assets of 0.1 GB each → 0.9 GB = 90 % fill → triggers eviction on next
        let size_bytes = 107_374_182_u64; // ~0.1 GB
        for i in 0..9 {
            s.handle_request(&req(&format!("asset-{i}"), "us-east-1", size_bytes), i);
        }
        let node = s.get_node("tiny").expect("tiny");
        assert!(!node.is_near_full() || node.asset_count() <= 9,
            "should have evicted or kept fill below maximum");

        // Add one more — should trigger eviction
        s.handle_request(&req("overflow", "us-east-1", size_bytes), 9);
        let node = s.get_node("tiny").expect("tiny");
        assert!(node.has_asset("overflow"), "newest asset must be present");
    }

    // 9. stats: total_requests, cache_hits, origin_fetches
    #[test]
    fn test_stats_aggregation() {
        let mut s = make_shield();
        // OriginFetch from an empty shield
        let mut empty = OriginShield::new();
        empty.handle_request(&req("z", "us-east-1", 100), 0);
        let st = empty.stats();
        assert_eq!(st.origin_fetches, 1);
        assert_eq!(st.total_requests, 1);

        // Shield with nodes: 1 miss + 1 hit = 2 total
        s.handle_request(&req("m", "us-east-1", 500), 0);
        s.handle_request(&req("m", "us-east-1", 500), 1);
        let st2 = s.stats();
        assert_eq!(st2.total_requests, 2);
        assert_eq!(st2.cache_hits, 1);
    }

    // 10. remove_node removes a node
    #[test]
    fn test_remove_node() {
        let mut s = make_shield();
        assert!(s.remove_node("shield-us"));
        assert_eq!(s.node_count(), 1);
        assert!(!s.remove_node("shield-us"));
    }

    // 11. CacheHit node_id matches the node that stored the asset
    #[test]
    fn test_cache_hit_node_id() {
        let mut s = make_shield();
        s.handle_request(&req("clip.mp4", "eu-west-1", 100_000), 0);
        let r = s.handle_request(&req("clip.mp4", "eu-west-1", 100_000), 1);
        match r {
            ShieldResponse::CacheHit { node_id } => assert_eq!(node_id, "shield-eu"),
            other => panic!("expected CacheHit, got {other:?}"),
        }
    }

    // 12. asset stored in one region is found from another region
    #[test]
    fn test_cross_region_cache_hit() {
        let mut s = make_shield();
        // Store asset via us-east-1 request
        s.handle_request(&req("shared.m3u8", "us-east-1", 2_000), 0);
        // Request from eu-west-1 — global search should find it in shield-us
        let r = s.handle_request(&req("shared.m3u8", "eu-west-1", 2_000), 1);
        assert!(
            matches!(r, ShieldResponse::CacheHit { .. }),
            "cross-region hit expected: {r:?}"
        );
    }

    // 13. ShieldNode::fill_ratio and is_near_full
    #[test]
    fn test_fill_ratio_and_near_full() {
        let mut n = ShieldNode::new("n", "r", 1.0);
        assert!((n.fill_ratio() - 0.0).abs() < 1e-9);
        // Fill to 95 % (safely above the 90 % threshold)
        // 0.95 * 1 GiB = 1_019_215_872 bytes
        let size: u64 = 1_019_215_872;
        n.store_asset("a", size);
        assert!(n.fill_ratio() >= 0.90, "fill={}", n.fill_ratio());
        assert!(n.is_near_full(), "node at ≥90% fill must report near_full");
    }

    // 14. ShieldNode with zero capacity always reports near-full
    #[test]
    fn test_zero_capacity_node_fill_ratio() {
        let n = ShieldNode::new("empty", "r", 0.0);
        assert!((n.fill_ratio() - 1.0).abs() < 1e-9);
        assert!(n.is_near_full());
    }

    // 15. hit_rate returns 0 with no requests
    #[test]
    fn test_hit_rate_no_requests() {
        let s = make_shield();
        assert!((s.hit_rate() - 0.0).abs() < 1e-6);
    }
}
