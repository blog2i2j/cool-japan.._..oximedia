//! Content delivery module with edge caching configuration for multi-region deployment.
//!
//! Implements:
//! - Edge node registry with geographic location metadata
//! - Cache policy definition (TTL, Vary headers, cache-control directives)
//! - CDN routing: origin-pull vs cache-hit decision
//! - Invalidation: single-key, prefix, and tag-based purge
//! - Geo-routing: select the nearest edge by Haversine distance
//! - Cache statistics: hit rate, bandwidth saved, origin offload ratio
//! - Replication state: track which edges have a cached copy

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// ── Geographic coordinates ─────────────────────────────────────────────────────

/// A geographic coordinate in degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoCoord {
    /// Latitude in degrees (-90 to 90).
    pub lat: f64,
    /// Longitude in degrees (-180 to 180).
    pub lon: f64,
}

impl GeoCoord {
    /// Creates a new coordinate.
    pub fn new(lat: f64, lon: f64) -> Self {
        Self { lat, lon }
    }

    /// Computes the Haversine distance to another coordinate in kilometres.
    pub fn distance_km(&self, other: &Self) -> f64 {
        const EARTH_RADIUS_KM: f64 = 6_371.0;
        let dlat = (other.lat - self.lat).to_radians();
        let dlon = (other.lon - self.lon).to_radians();
        let a = (dlat / 2.0).sin().powi(2)
            + self.lat.to_radians().cos()
                * other.lat.to_radians().cos()
                * (dlon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
        EARTH_RADIUS_KM * c
    }
}

// ── Edge node ──────────────────────────────────────────────────────────────────

/// Status of an edge node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeNodeStatus {
    /// Fully operational.
    Online,
    /// Partially degraded (e.g. reduced capacity).
    Degraded,
    /// Not accepting traffic.
    Offline,
    /// Warming up (cache empty, high origin pull expected).
    WarmingUp,
}

impl EdgeNodeStatus {
    /// Returns `true` if the edge can serve traffic.
    pub fn is_serving(self) -> bool {
        matches!(self, Self::Online | Self::Degraded | Self::WarmingUp)
    }

    /// Label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Online => "online",
            Self::Degraded => "degraded",
            Self::Offline => "offline",
            Self::WarmingUp => "warming_up",
        }
    }
}

/// An edge node in the content delivery network.
#[derive(Debug, Clone)]
pub struct EdgeNode {
    /// Unique node identifier (e.g. `edge-us-east-1`).
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Geographic location.
    pub location: GeoCoord,
    /// Region label (e.g. `us-east`, `eu-west`).
    pub region: String,
    /// Status.
    pub status: EdgeNodeStatus,
    /// Maximum cache capacity in bytes.
    pub cache_capacity_bytes: u64,
    /// Current cache used in bytes.
    pub cache_used_bytes: u64,
    /// Total cache hits served.
    pub cache_hits: u64,
    /// Total cache misses (origin pulls).
    pub cache_misses: u64,
    /// Total bytes served from cache.
    pub bytes_served_from_cache: u64,
    /// Total bytes pulled from origin.
    pub bytes_pulled_from_origin: u64,
}

impl EdgeNode {
    /// Creates a new edge node.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        location: GeoCoord,
        region: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            location,
            region: region.into(),
            status: EdgeNodeStatus::Online,
            cache_capacity_bytes: 100 * 1024 * 1024 * 1024, // 100 GB default
            cache_used_bytes: 0,
            cache_hits: 0,
            cache_misses: 0,
            bytes_served_from_cache: 0,
            bytes_pulled_from_origin: 0,
        }
    }

    /// Sets the cache capacity.
    pub fn with_capacity(mut self, bytes: u64) -> Self {
        self.cache_capacity_bytes = bytes;
        self
    }

    /// Cache fill ratio (0.0 – 1.0).
    pub fn cache_fill_ratio(&self) -> f64 {
        if self.cache_capacity_bytes == 0 {
            return 0.0;
        }
        self.cache_used_bytes as f64 / self.cache_capacity_bytes as f64
    }

    /// Cache hit rate (0.0 – 1.0).
    pub fn hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            return 0.0;
        }
        self.cache_hits as f64 / total as f64
    }

    /// Bandwidth saving ratio (cache bytes / total bytes).
    pub fn bandwidth_saving(&self) -> f64 {
        let total = self.bytes_served_from_cache + self.bytes_pulled_from_origin;
        if total == 0 {
            return 0.0;
        }
        self.bytes_served_from_cache as f64 / total as f64
    }
}

// ── Cache policy ───────────────────────────────────────────────────────────────

/// Cache directive for the `Cache-Control` header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheDirective {
    /// `public`: Response may be cached by shared caches.
    Public,
    /// `private`: Response is intended for a single user.
    Private,
    /// `no-cache`: Must revalidate with origin before serving from cache.
    NoCache,
    /// `no-store`: Must not cache at all.
    NoStore,
    /// `must-revalidate`: Stale responses must be revalidated.
    MustRevalidate,
    /// `s-maxage=N`: Overrides `max-age` for shared caches.
    SMaxAge(u64),
    /// `stale-while-revalidate=N`: Serve stale while fetching fresh.
    StaleWhileRevalidate(u64),
}

impl CacheDirective {
    /// Serializes the directive to a string.
    pub fn to_header_value(&self) -> String {
        match self {
            Self::Public => "public".to_string(),
            Self::Private => "private".to_string(),
            Self::NoCache => "no-cache".to_string(),
            Self::NoStore => "no-store".to_string(),
            Self::MustRevalidate => "must-revalidate".to_string(),
            Self::SMaxAge(n) => format!("s-maxage={}", n),
            Self::StaleWhileRevalidate(n) => format!("stale-while-revalidate={}", n),
        }
    }
}

/// A cache policy applied to a set of URL patterns.
#[derive(Debug, Clone)]
pub struct CachePolicy {
    /// Policy name.
    pub name: String,
    /// URL path patterns this policy applies to (prefix match).
    pub path_patterns: Vec<String>,
    /// TTL at the edge.
    pub ttl: Duration,
    /// Browser TTL (for `max-age`).
    pub browser_ttl: Duration,
    /// Cache-Control directives.
    pub directives: Vec<CacheDirective>,
    /// Vary header values.
    pub vary_headers: Vec<String>,
    /// Cache tags for grouped invalidation.
    pub cache_tags: Vec<String>,
    /// Whether to bypass cache for authenticated requests.
    pub bypass_for_auth: bool,
    /// Whether query string affects caching.
    pub respect_query_string: bool,
}

impl CachePolicy {
    /// Creates a new cache policy.
    pub fn new(name: impl Into<String>, ttl: Duration) -> Self {
        Self {
            name: name.into(),
            path_patterns: Vec::new(),
            ttl,
            browser_ttl: ttl,
            directives: vec![CacheDirective::Public],
            vary_headers: Vec::new(),
            cache_tags: Vec::new(),
            bypass_for_auth: false,
            respect_query_string: true,
        }
    }

    /// Adds a path pattern.
    pub fn with_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.path_patterns.push(pattern.into());
        self
    }

    /// Sets the browser TTL.
    pub fn with_browser_ttl(mut self, ttl: Duration) -> Self {
        self.browser_ttl = ttl;
        self
    }

    /// Adds a cache directive.
    pub fn with_directive(mut self, directive: CacheDirective) -> Self {
        self.directives.push(directive);
        self
    }

    /// Adds a Vary header.
    pub fn vary_on(mut self, header: impl Into<String>) -> Self {
        self.vary_headers.push(header.into());
        self
    }

    /// Adds a cache tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.cache_tags.push(tag.into());
        self
    }

    /// Enables bypass for authenticated requests.
    pub fn bypass_auth(mut self) -> Self {
        self.bypass_for_auth = true;
        self
    }

    /// Checks whether this policy applies to the given path.
    pub fn applies_to(&self, path: &str) -> bool {
        if self.path_patterns.is_empty() {
            return true; // no patterns = universal
        }
        self.path_patterns
            .iter()
            .any(|p| path.starts_with(p.as_str()))
    }

    /// Builds the `Cache-Control` header value.
    pub fn cache_control_header(&self) -> String {
        let mut parts = vec![format!("max-age={}", self.browser_ttl.as_secs())];
        for d in &self.directives {
            parts.push(d.to_header_value());
        }
        parts.join(", ")
    }
}

// ── Cache entry ────────────────────────────────────────────────────────────────

/// A cached item at an edge node.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Cache key (URL + vary hash).
    pub key: String,
    /// Size in bytes.
    pub size_bytes: u64,
    /// When the entry was cached.
    pub cached_at: Instant,
    /// Time-to-live.
    pub ttl: Duration,
    /// Cache tags for grouped invalidation.
    pub tags: Vec<String>,
    /// ETag value.
    pub etag: Option<String>,
    /// Content-Type.
    pub content_type: String,
}

impl CacheEntry {
    /// Creates a new cache entry.
    pub fn new(key: impl Into<String>, size_bytes: u64, ttl: Duration) -> Self {
        Self {
            key: key.into(),
            size_bytes,
            cached_at: Instant::now(),
            ttl,
            tags: Vec::new(),
            etag: None,
            content_type: "application/octet-stream".to_string(),
        }
    }

    /// Returns `true` if the entry is still fresh.
    pub fn is_fresh(&self) -> bool {
        self.cached_at.elapsed() < self.ttl
    }

    /// Returns the age of the entry.
    pub fn age(&self) -> Duration {
        self.cached_at.elapsed()
    }

    /// Returns the remaining TTL.
    pub fn remaining_ttl(&self) -> Duration {
        self.ttl.saturating_sub(self.cached_at.elapsed())
    }

    /// Adds a cache tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Sets the ETag.
    pub fn with_etag(mut self, etag: impl Into<String>) -> Self {
        self.etag = Some(etag.into());
        self
    }

    /// Sets the content type.
    pub fn with_content_type(mut self, ct: impl Into<String>) -> Self {
        self.content_type = ct.into();
        self
    }
}

// ── Edge cache ─────────────────────────────────────────────────────────────────

/// Edge-local cache store (simplified LRU approximation via insertion-order eviction).
pub struct EdgeCache {
    /// Cached entries keyed by cache key.
    entries: HashMap<String, CacheEntry>,
    /// Capacity in bytes.
    capacity_bytes: u64,
    /// Current size in bytes.
    current_size_bytes: u64,
    /// Insertion-order tracking for eviction.
    insertion_order: Vec<String>,
}

impl EdgeCache {
    /// Creates a new edge cache.
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            entries: HashMap::new(),
            capacity_bytes,
            current_size_bytes: 0,
            insertion_order: Vec::new(),
        }
    }

    /// Stores an entry, evicting oldest if necessary.
    pub fn put(&mut self, entry: CacheEntry) {
        // Evict if needed
        while self.current_size_bytes + entry.size_bytes > self.capacity_bytes
            && !self.insertion_order.is_empty()
        {
            let oldest_key = self.insertion_order.remove(0);
            if let Some(old) = self.entries.remove(&oldest_key) {
                self.current_size_bytes = self.current_size_bytes.saturating_sub(old.size_bytes);
            }
        }

        self.current_size_bytes += entry.size_bytes;
        self.insertion_order.push(entry.key.clone());
        self.entries.insert(entry.key.clone(), entry);
    }

    /// Retrieves a fresh entry (returns `None` if stale or absent).
    pub fn get(&self, key: &str) -> Option<&CacheEntry> {
        self.entries.get(key).filter(|e| e.is_fresh())
    }

    /// Invalidates a single key.
    pub fn invalidate(&mut self, key: &str) -> bool {
        if let Some(entry) = self.entries.remove(key) {
            self.current_size_bytes = self.current_size_bytes.saturating_sub(entry.size_bytes);
            self.insertion_order.retain(|k| k != key);
            true
        } else {
            false
        }
    }

    /// Invalidates all keys with the given prefix.
    pub fn invalidate_prefix(&mut self, prefix: &str) -> usize {
        let keys: Vec<String> = self
            .entries
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();
        let count = keys.len();
        for k in keys {
            self.invalidate(&k);
        }
        count
    }

    /// Invalidates all entries with a given cache tag.
    pub fn invalidate_tag(&mut self, tag: &str) -> usize {
        let keys: Vec<String> = self
            .entries
            .values()
            .filter(|e| e.tags.iter().any(|t| t == tag))
            .map(|e| e.key.clone())
            .collect();
        let count = keys.len();
        for k in keys {
            self.invalidate(&k);
        }
        count
    }

    /// Purges all stale entries.
    pub fn purge_stale(&mut self) -> usize {
        let stale_keys: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, e)| !e.is_fresh())
            .map(|(k, _)| k.clone())
            .collect();
        let count = stale_keys.len();
        for k in stale_keys {
            self.invalidate(&k);
        }
        count
    }

    /// Number of entries in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Current size in bytes.
    pub fn current_size_bytes(&self) -> u64 {
        self.current_size_bytes
    }

    /// Fill ratio.
    pub fn fill_ratio(&self) -> f64 {
        if self.capacity_bytes == 0 {
            return 0.0;
        }
        self.current_size_bytes as f64 / self.capacity_bytes as f64
    }
}

// ── Replication state ──────────────────────────────────────────────────────────

/// Tracks which edge nodes have a cached copy of a given key.
#[derive(Debug, Clone, Default)]
pub struct ReplicationState {
    /// key → set of edge node IDs that have a cached copy.
    replicated: HashMap<String, HashSet<String>>,
}

impl ReplicationState {
    /// Creates a new replication state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records that `edge_id` now has a cached copy of `key`.
    pub fn mark_replicated(&mut self, key: impl Into<String>, edge_id: impl Into<String>) {
        self.replicated
            .entry(key.into())
            .or_default()
            .insert(edge_id.into());
    }

    /// Returns the set of edge IDs that have a copy of `key`.
    pub fn edges_with(&self, key: &str) -> HashSet<&str> {
        self.replicated
            .get(key)
            .map(|s| s.iter().map(String::as_str).collect())
            .unwrap_or_default()
    }

    /// Removes a key from the replication tracking (e.g. after invalidation).
    pub fn clear_key(&mut self, key: &str) {
        self.replicated.remove(key);
    }

    /// Returns the total number of tracked keys.
    pub fn tracked_keys(&self) -> usize {
        self.replicated.len()
    }

    /// Returns the replication count for a key.
    pub fn replication_count(&self, key: &str) -> usize {
        self.replicated.get(key).map(|s| s.len()).unwrap_or(0)
    }
}

// ── CDN manager ────────────────────────────────────────────────────────────────

/// Configuration for the content delivery system.
#[derive(Debug, Clone)]
pub struct ContentDeliveryConfig {
    /// Default TTL for unconfigured paths.
    pub default_ttl: Duration,
    /// Whether to serve stale content while origin is down.
    pub serve_stale_on_origin_error: bool,
    /// Whether to collapse concurrent origin requests for the same key.
    pub request_coalescing: bool,
    /// Whether to stream edge metrics to a monitoring sink.
    pub metrics_enabled: bool,
    /// Minimum replication factor (number of edges that must cache a key).
    pub min_replication_factor: usize,
}

impl Default for ContentDeliveryConfig {
    fn default() -> Self {
        Self {
            default_ttl: Duration::from_secs(3600),
            serve_stale_on_origin_error: true,
            request_coalescing: true,
            metrics_enabled: true,
            min_replication_factor: 1,
        }
    }
}

/// Global CDN statistics snapshot.
#[derive(Debug, Clone, Default)]
pub struct CdnStats {
    /// Total requests handled.
    pub total_requests: u64,
    /// Requests served from cache (aggregated across all edges).
    pub cache_hits: u64,
    /// Requests that required an origin pull.
    pub cache_misses: u64,
    /// Bytes served from cache.
    pub bytes_from_cache: u64,
    /// Bytes pulled from origin.
    pub bytes_from_origin: u64,
    /// Total invalidation operations.
    pub invalidations: u64,
}

impl CdnStats {
    /// Overall hit rate.
    pub fn hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            return 0.0;
        }
        self.cache_hits as f64 / total as f64
    }

    /// Origin offload ratio (bytes saved from origin).
    pub fn origin_offload(&self) -> f64 {
        let total = self.bytes_from_cache + self.bytes_from_origin;
        if total == 0 {
            return 0.0;
        }
        self.bytes_from_cache as f64 / total as f64
    }
}

/// The content delivery manager.
pub struct ContentDeliveryManager {
    config: ContentDeliveryConfig,
    /// Registered edge nodes.
    edges: HashMap<String, EdgeNode>,
    /// Cache policies.
    policies: Vec<CachePolicy>,
    /// Per-edge cache stores.
    edge_caches: HashMap<String, EdgeCache>,
    /// Replication tracking.
    replication: ReplicationState,
    /// Global stats.
    stats: CdnStats,
    /// Timestamp generation counter (for deterministic ETag generation in tests).
    etag_counter: u64,
}

impl ContentDeliveryManager {
    /// Creates a new manager.
    pub fn new(config: ContentDeliveryConfig) -> Self {
        Self {
            config,
            edges: HashMap::new(),
            policies: Vec::new(),
            edge_caches: HashMap::new(),
            replication: ReplicationState::new(),
            stats: CdnStats::default(),
            etag_counter: 0,
        }
    }

    /// Registers an edge node.
    pub fn register_edge(&mut self, edge: EdgeNode) {
        let capacity = edge.cache_capacity_bytes;
        let id = edge.id.clone();
        self.edges.insert(id.clone(), edge);
        self.edge_caches.insert(id, EdgeCache::new(capacity));
    }

    /// Registers a cache policy.
    pub fn register_policy(&mut self, policy: CachePolicy) {
        self.policies.push(policy);
    }

    /// Returns the most specific cache policy for a path.
    pub fn policy_for(&self, path: &str) -> Option<&CachePolicy> {
        // Policies with patterns take precedence; find longest matching prefix.
        let mut best: Option<&CachePolicy> = None;
        let mut best_len = 0usize;
        for policy in &self.policies {
            if policy.applies_to(path) {
                let match_len = policy
                    .path_patterns
                    .iter()
                    .filter(|p| path.starts_with(p.as_str()))
                    .map(|p| p.len())
                    .max()
                    .unwrap_or(0);
                if match_len >= best_len {
                    best_len = match_len;
                    best = Some(policy);
                }
            }
        }
        best
    }

    /// Selects the nearest edge to a client location.
    pub fn nearest_edge(&self, client: &GeoCoord) -> Option<&EdgeNode> {
        self.edges
            .values()
            .filter(|e| e.status.is_serving())
            .min_by(|a, b| {
                let da = a.location.distance_km(client);
                let db = b.location.distance_km(client);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Attempts to serve a request from the edge cache.
    ///
    /// Returns `Some(entry)` on a cache hit, `None` on a miss.
    pub fn serve_from_cache(&mut self, edge_id: &str, key: &str) -> Option<CacheEntry> {
        let edge = self.edges.get_mut(edge_id)?;
        let cache = self.edge_caches.get(edge_id)?;

        self.stats.total_requests += 1;

        if let Some(entry) = cache.get(key) {
            edge.cache_hits += 1;
            edge.bytes_served_from_cache += entry.size_bytes;
            self.stats.cache_hits += 1;
            self.stats.bytes_from_cache += entry.size_bytes;
            Some(entry.clone())
        } else {
            edge.cache_misses += 1;
            self.stats.cache_misses += 1;
            None
        }
    }

    /// Caches a response at an edge node.
    pub fn cache_at_edge(&mut self, edge_id: &str, entry: CacheEntry) -> bool {
        if let Some(cache) = self.edge_caches.get_mut(edge_id) {
            let key = entry.key.clone();
            let size = entry.size_bytes;
            cache.put(entry);
            if let Some(edge) = self.edges.get_mut(edge_id) {
                edge.cache_used_bytes = edge.cache_used_bytes.saturating_add(size);
                edge.bytes_pulled_from_origin += size;
                self.stats.bytes_from_origin += size;
            }
            self.replication.mark_replicated(key, edge_id);
            true
        } else {
            false
        }
    }

    /// Invalidates a key across all edges.
    pub fn invalidate_key(&mut self, key: &str) -> usize {
        let mut total = 0usize;
        for cache in self.edge_caches.values_mut() {
            if cache.invalidate(key) {
                total += 1;
            }
        }
        self.replication.clear_key(key);
        self.stats.invalidations += 1;
        total
    }

    /// Invalidates all keys with a prefix across all edges.
    pub fn invalidate_prefix(&mut self, prefix: &str) -> usize {
        let mut total = 0usize;
        for cache in self.edge_caches.values_mut() {
            total += cache.invalidate_prefix(prefix);
        }
        self.stats.invalidations += 1;
        total
    }

    /// Invalidates all keys with a cache tag across all edges.
    pub fn invalidate_tag(&mut self, tag: &str) -> usize {
        let mut total = 0usize;
        for cache in self.edge_caches.values_mut() {
            total += cache.invalidate_tag(tag);
        }
        self.stats.invalidations += 1;
        total
    }

    /// Updates the status of an edge node.
    pub fn set_edge_status(&mut self, edge_id: &str, status: EdgeNodeStatus) {
        if let Some(edge) = self.edges.get_mut(edge_id) {
            edge.status = status;
        }
    }

    /// Returns global CDN statistics.
    pub fn stats(&self) -> &CdnStats {
        &self.stats
    }

    /// Returns an edge node by ID.
    pub fn edge(&self, id: &str) -> Option<&EdgeNode> {
        self.edges.get(id)
    }

    /// Returns all edge nodes.
    pub fn edges(&self) -> &HashMap<String, EdgeNode> {
        &self.edges
    }

    /// Returns the number of online edges.
    pub fn online_edge_count(&self) -> usize {
        self.edges
            .values()
            .filter(|e| e.status == EdgeNodeStatus::Online)
            .count()
    }

    /// Generates a simple ETag for a content blob.
    pub fn generate_etag(&mut self, content_hash: u64) -> String {
        self.etag_counter += 1;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        format!("\"W/{:x}-{:x}\"", now ^ content_hash, self.etag_counter)
    }
}

impl Default for ContentDeliveryManager {
    fn default() -> Self {
        Self::new(ContentDeliveryConfig::default())
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager() -> ContentDeliveryManager {
        let mut mgr = ContentDeliveryManager::default();

        mgr.register_edge(
            EdgeNode::new(
                "edge-us",
                "US East",
                GeoCoord::new(40.7128, -74.006), // New York
                "us-east",
            )
            .with_capacity(10 * 1024 * 1024), // 10 MB for tests
        );
        mgr.register_edge(
            EdgeNode::new(
                "edge-eu",
                "EU West",
                GeoCoord::new(51.5074, -0.1278), // London
                "eu-west",
            )
            .with_capacity(10 * 1024 * 1024),
        );

        mgr.register_policy(
            CachePolicy::new("media-policy", Duration::from_secs(86400))
                .with_pattern("/media/")
                .with_tag("media"),
        );

        mgr
    }

    // GeoCoord tests

    #[test]
    fn test_distance_km_same_point() {
        let p = GeoCoord::new(48.8566, 2.3522);
        assert!(p.distance_km(&p) < 0.001);
    }

    #[test]
    fn test_distance_km_known() {
        // New York to London ≈ 5570 km
        let ny = GeoCoord::new(40.7128, -74.006);
        let lon = GeoCoord::new(51.5074, -0.1278);
        let d = ny.distance_km(&lon);
        assert!(d > 5000.0 && d < 6000.0, "distance = {}", d);
    }

    // EdgeNodeStatus tests

    #[test]
    fn test_edge_status_is_serving() {
        assert!(EdgeNodeStatus::Online.is_serving());
        assert!(EdgeNodeStatus::Degraded.is_serving());
        assert!(EdgeNodeStatus::WarmingUp.is_serving());
        assert!(!EdgeNodeStatus::Offline.is_serving());
    }

    #[test]
    fn test_edge_status_labels() {
        assert_eq!(EdgeNodeStatus::Online.label(), "online");
        assert_eq!(EdgeNodeStatus::Offline.label(), "offline");
    }

    // EdgeNode tests

    #[test]
    fn test_edge_node_hit_rate_zero() {
        let e = EdgeNode::new("e1", "E1", GeoCoord::new(0.0, 0.0), "us");
        assert!((e.hit_rate()).abs() < 1e-9);
    }

    #[test]
    fn test_edge_node_hit_rate_calculated() {
        let mut e = EdgeNode::new("e1", "E1", GeoCoord::new(0.0, 0.0), "us");
        e.cache_hits = 7;
        e.cache_misses = 3;
        assert!((e.hit_rate() - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_edge_node_bandwidth_saving() {
        let mut e = EdgeNode::new("e1", "E1", GeoCoord::new(0.0, 0.0), "us");
        e.bytes_served_from_cache = 900;
        e.bytes_pulled_from_origin = 100;
        assert!((e.bandwidth_saving() - 0.9).abs() < 1e-9);
    }

    // CacheDirective tests

    #[test]
    fn test_cache_directive_header_values() {
        assert_eq!(CacheDirective::Public.to_header_value(), "public");
        assert_eq!(
            CacheDirective::SMaxAge(3600).to_header_value(),
            "s-maxage=3600"
        );
        assert_eq!(
            CacheDirective::StaleWhileRevalidate(60).to_header_value(),
            "stale-while-revalidate=60"
        );
    }

    // CachePolicy tests

    #[test]
    fn test_cache_policy_applies_to() {
        let p = CachePolicy::new("p", Duration::from_secs(60)).with_pattern("/media/");
        assert!(p.applies_to("/media/video.webm"));
        assert!(!p.applies_to("/api/v1/media"));
    }

    #[test]
    fn test_cache_policy_no_patterns_universal() {
        let p = CachePolicy::new("p", Duration::from_secs(60));
        assert!(p.applies_to("/anything"));
    }

    #[test]
    fn test_cache_control_header() {
        let p = CachePolicy::new("p", Duration::from_secs(3600))
            .with_directive(CacheDirective::SMaxAge(7200));
        let header = p.cache_control_header();
        assert!(header.contains("max-age=3600"));
        assert!(header.contains("public"));
        assert!(header.contains("s-maxage=7200"));
    }

    // CacheEntry tests

    #[test]
    fn test_cache_entry_is_fresh() {
        let entry = CacheEntry::new("/media/m1.webm", 1024, Duration::from_secs(3600));
        assert!(entry.is_fresh());
    }

    #[test]
    fn test_cache_entry_stale() {
        // TTL of 0 = immediately stale
        let entry = CacheEntry::new("/media/m1.webm", 1024, Duration::ZERO);
        // Give a tiny bit of time to ensure it's expired
        std::thread::sleep(Duration::from_millis(1));
        assert!(!entry.is_fresh());
    }

    // EdgeCache tests

    #[test]
    fn test_edge_cache_put_and_get() {
        let mut cache = EdgeCache::new(1024 * 1024);
        let entry = CacheEntry::new("/media/m1.webm", 512, Duration::from_secs(3600));
        cache.put(entry);
        assert!(cache.get("/media/m1.webm").is_some());
    }

    #[test]
    fn test_edge_cache_invalidate() {
        let mut cache = EdgeCache::new(1024 * 1024);
        cache.put(CacheEntry::new(
            "/media/m1.webm",
            100,
            Duration::from_secs(3600),
        ));
        assert!(cache.invalidate("/media/m1.webm"));
        assert!(cache.get("/media/m1.webm").is_none());
    }

    #[test]
    fn test_edge_cache_invalidate_prefix() {
        let mut cache = EdgeCache::new(1024 * 1024);
        cache.put(CacheEntry::new(
            "/media/a.webm",
            100,
            Duration::from_secs(3600),
        ));
        cache.put(CacheEntry::new(
            "/media/b.webm",
            100,
            Duration::from_secs(3600),
        ));
        cache.put(CacheEntry::new(
            "/other/c.webm",
            100,
            Duration::from_secs(3600),
        ));
        let removed = cache.invalidate_prefix("/media/");
        assert_eq!(removed, 2);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_edge_cache_invalidate_tag() {
        let mut cache = EdgeCache::new(1024 * 1024);
        cache.put(
            CacheEntry::new("/media/a.webm", 100, Duration::from_secs(3600)).with_tag("media"),
        );
        cache.put(
            CacheEntry::new("/other/b.webm", 100, Duration::from_secs(3600)).with_tag("other"),
        );
        let removed = cache.invalidate_tag("media");
        assert_eq!(removed, 1);
    }

    #[test]
    fn test_edge_cache_eviction_on_overflow() {
        let mut cache = EdgeCache::new(200); // only 200 bytes
        cache.put(CacheEntry::new("k1", 100, Duration::from_secs(3600)));
        cache.put(CacheEntry::new("k2", 100, Duration::from_secs(3600)));
        // Third entry overflows; k1 should be evicted
        cache.put(CacheEntry::new("k3", 100, Duration::from_secs(3600)));
        assert!(cache.len() <= 2);
        assert!(cache.get("k3").is_some(), "newest entry should be present");
    }

    // ReplicationState tests

    #[test]
    fn test_replication_state_tracking() {
        let mut state = ReplicationState::new();
        state.mark_replicated("/media/m1.webm", "edge-us");
        state.mark_replicated("/media/m1.webm", "edge-eu");
        assert_eq!(state.replication_count("/media/m1.webm"), 2);
    }

    #[test]
    fn test_replication_state_clear_key() {
        let mut state = ReplicationState::new();
        state.mark_replicated("/media/m1.webm", "edge-us");
        state.clear_key("/media/m1.webm");
        assert_eq!(state.replication_count("/media/m1.webm"), 0);
    }

    // ContentDeliveryManager tests

    #[test]
    fn test_nearest_edge() {
        let mgr = make_manager();
        // A client in New York should be routed to edge-us
        let client = GeoCoord::new(40.7128, -74.006);
        let nearest = mgr.nearest_edge(&client).expect("should find an edge");
        assert_eq!(nearest.id, "edge-us");
    }

    #[test]
    fn test_cache_miss_then_hit() {
        let mut mgr = make_manager();
        // First request = miss
        assert!(mgr.serve_from_cache("edge-us", "/media/m1.webm").is_none());
        // Cache the entry
        let entry = CacheEntry::new("/media/m1.webm", 1024, Duration::from_secs(3600));
        assert!(mgr.cache_at_edge("edge-us", entry));
        // Second request = hit
        assert!(mgr.serve_from_cache("edge-us", "/media/m1.webm").is_some());
    }

    #[test]
    fn test_invalidate_key_across_edges() {
        let mut mgr = make_manager();
        let entry_us = CacheEntry::new("/media/m1.webm", 100, Duration::from_secs(3600));
        let entry_eu = CacheEntry::new("/media/m1.webm", 100, Duration::from_secs(3600));
        mgr.cache_at_edge("edge-us", entry_us);
        mgr.cache_at_edge("edge-eu", entry_eu);

        let removed = mgr.invalidate_key("/media/m1.webm");
        assert_eq!(removed, 2);

        assert!(mgr.serve_from_cache("edge-us", "/media/m1.webm").is_none());
        assert!(mgr.serve_from_cache("edge-eu", "/media/m1.webm").is_none());
    }

    #[test]
    fn test_cdn_stats_hit_rate() {
        let mut mgr = make_manager();
        // Simulate a miss + cache + hit
        mgr.serve_from_cache("edge-us", "/media/m1.webm"); // miss
        let entry = CacheEntry::new("/media/m1.webm", 100, Duration::from_secs(3600));
        mgr.cache_at_edge("edge-us", entry);
        mgr.serve_from_cache("edge-us", "/media/m1.webm"); // hit

        let stats = mgr.stats();
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
        assert!((stats.hit_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_online_edge_count() {
        let mut mgr = make_manager();
        assert_eq!(mgr.online_edge_count(), 2);
        mgr.set_edge_status("edge-us", EdgeNodeStatus::Offline);
        assert_eq!(mgr.online_edge_count(), 1);
    }

    #[test]
    fn test_generate_etag_unique() {
        let mut mgr = make_manager();
        let t1 = mgr.generate_etag(0xDEAD_BEEF);
        let t2 = mgr.generate_etag(0xDEAD_BEEF);
        assert_ne!(t1, t2); // counter increments
    }
}
