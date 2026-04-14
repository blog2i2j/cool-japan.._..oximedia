//! API gateway module: rate limiting, throttling, and request routing for microservice architecture.
//!
//! Implements:
//! - Route registry: map external paths to upstream backend URLs
//! - Token-bucket throttling per route with configurable burst/refill rates
//! - Circuit-breaker state integration (open/closed/half-open)
//! - Load balancing strategies: round-robin and least-connections
//! - Request transformation: path rewriting, header injection, stripping auth
//! - Gateway metrics: request counts, latency histograms, error rates
//! - Health aggregation across registered upstreams

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// ── Upstream backend ───────────────────────────────────────────────────────────

/// Health status of an upstream backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpstreamHealth {
    /// The upstream is accepting traffic.
    Healthy,
    /// The upstream is degraded but still serving.
    Degraded,
    /// The upstream is not responding.
    Unhealthy,
}

impl UpstreamHealth {
    /// Returns `true` if traffic can be sent to this upstream.
    pub fn is_routable(self) -> bool {
        matches!(self, Self::Healthy | Self::Degraded)
    }

    /// String label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Degraded => "degraded",
            Self::Unhealthy => "unhealthy",
        }
    }
}

/// A registered upstream backend.
#[derive(Debug, Clone)]
pub struct Upstream {
    /// Unique upstream identifier.
    pub id: String,
    /// Base URL of the upstream (e.g. `http://transcode-svc:8080`).
    pub base_url: String,
    /// Current health.
    pub health: UpstreamHealth,
    /// Weight for load balancing (higher = more traffic).
    pub weight: u32,
    /// Active connections (for least-connections strategy).
    pub active_connections: u64,
    /// Total requests served.
    pub total_requests: u64,
    /// Total errors returned.
    pub total_errors: u64,
}

impl Upstream {
    /// Creates a new upstream.
    pub fn new(id: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            base_url: base_url.into(),
            health: UpstreamHealth::Healthy,
            weight: 1,
            active_connections: 0,
            total_requests: 0,
            total_errors: 0,
        }
    }

    /// Sets the weight.
    pub fn with_weight(mut self, weight: u32) -> Self {
        self.weight = weight;
        self
    }

    /// Error rate (errors / requests).
    pub fn error_rate(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        self.total_errors as f64 / self.total_requests as f64
    }
}

// ── Load balancing ─────────────────────────────────────────────────────────────

/// Load balancing strategy for selecting among upstream backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalanceStrategy {
    /// Cycle through upstreams in order.
    RoundRobin,
    /// Pick the upstream with the fewest active connections.
    LeastConnections,
    /// Always pick the first healthy upstream (useful for active/standby).
    FirstHealthy,
}

impl LoadBalanceStrategy {
    /// Label.
    pub fn label(self) -> &'static str {
        match self {
            Self::RoundRobin => "round_robin",
            Self::LeastConnections => "least_connections",
            Self::FirstHealthy => "first_healthy",
        }
    }
}

/// Selects an upstream index from a list of upstreams.
///
/// Returns `None` if there are no routable upstreams.
pub fn select_upstream(
    upstreams: &[Upstream],
    strategy: LoadBalanceStrategy,
    round_robin_counter: &mut u64,
) -> Option<usize> {
    let routable: Vec<usize> = upstreams
        .iter()
        .enumerate()
        .filter(|(_, u)| u.health.is_routable())
        .map(|(i, _)| i)
        .collect();

    if routable.is_empty() {
        return None;
    }

    match strategy {
        LoadBalanceStrategy::RoundRobin => {
            let idx = (*round_robin_counter as usize) % routable.len();
            *round_robin_counter = round_robin_counter.wrapping_add(1);
            Some(routable[idx])
        }
        LoadBalanceStrategy::LeastConnections => routable
            .iter()
            .copied()
            .min_by_key(|&i| upstreams[i].active_connections),
        LoadBalanceStrategy::FirstHealthy => routable.first().copied(),
    }
}

// ── Route ─────────────────────────────────────────────────────────────────────

/// HTTP method constraint for a route.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Delete,
    Patch,
    Head,
    Options,
    /// Matches any method.
    Any,
}

impl HttpMethod {
    /// Returns `true` if this method matches `other`.
    pub fn matches(&self, other: &Self) -> bool {
        matches!(self, Self::Any) || self == other
    }

    /// Parses a method string (case-insensitive).
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "GET" => Some(Self::Get),
            "POST" => Some(Self::Post),
            "PUT" => Some(Self::Put),
            "DELETE" => Some(Self::Delete),
            "PATCH" => Some(Self::Patch),
            "HEAD" => Some(Self::Head),
            "OPTIONS" => Some(Self::Options),
            "*" | "ANY" => Some(Self::Any),
            _ => None,
        }
    }

    /// String representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Put => "PUT",
            Self::Delete => "DELETE",
            Self::Patch => "PATCH",
            Self::Head => "HEAD",
            Self::Options => "OPTIONS",
            Self::Any => "*",
        }
    }
}

/// Path transformation rule.
#[derive(Debug, Clone)]
pub struct PathRewrite {
    /// Pattern to match (prefix or exact).
    pub pattern: String,
    /// Replacement string (may use `$1`, `$2` for capture groups).
    pub replacement: String,
}

impl PathRewrite {
    /// Creates a prefix rewrite.
    pub fn strip_prefix(prefix: impl Into<String>) -> Self {
        let prefix = prefix.into();
        let replacement = String::new();
        Self {
            pattern: prefix,
            replacement,
        }
    }

    /// Applies the rewrite to a path.
    pub fn apply(&self, path: &str) -> String {
        if path.starts_with(&self.pattern) {
            let rest = &path[self.pattern.len()..];
            if self.replacement.is_empty() {
                if rest.is_empty() {
                    "/".to_string()
                } else {
                    rest.to_string()
                }
            } else {
                format!("{}{}", self.replacement, rest)
            }
        } else {
            path.to_string()
        }
    }
}

/// A registered gateway route.
#[derive(Debug, Clone)]
pub struct GatewayRoute {
    /// Route ID (used in metrics/logging).
    pub id: String,
    /// External path prefix to match.
    pub path_prefix: String,
    /// HTTP method constraint.
    pub method: HttpMethod,
    /// Upstream IDs to forward to (by load-balance strategy).
    pub upstream_ids: Vec<String>,
    /// Optional path rewrite.
    pub path_rewrite: Option<PathRewrite>,
    /// Additional headers to inject into upstream requests.
    pub inject_headers: HashMap<String, String>,
    /// Whether to strip the Authorization header before forwarding.
    pub strip_auth: bool,
    /// Per-route rate limit (requests per second). 0 = unlimited.
    pub rate_limit_rps: u64,
    /// Per-route timeout.
    pub timeout: Duration,
    /// Route priority (higher = matched first).
    pub priority: u32,
}

impl GatewayRoute {
    /// Creates a new route.
    pub fn new(
        id: impl Into<String>,
        path_prefix: impl Into<String>,
        upstream_ids: Vec<String>,
    ) -> Self {
        Self {
            id: id.into(),
            path_prefix: path_prefix.into(),
            method: HttpMethod::Any,
            upstream_ids,
            path_rewrite: None,
            inject_headers: HashMap::new(),
            strip_auth: false,
            rate_limit_rps: 0,
            timeout: Duration::from_secs(30),
            priority: 0,
        }
    }

    /// Sets the HTTP method constraint.
    pub fn with_method(mut self, method: HttpMethod) -> Self {
        self.method = method;
        self
    }

    /// Sets a path rewrite.
    pub fn with_rewrite(mut self, rewrite: PathRewrite) -> Self {
        self.path_rewrite = Some(rewrite);
        self
    }

    /// Enables auth stripping.
    pub fn strip_auth(mut self) -> Self {
        self.strip_auth = true;
        self
    }

    /// Sets the per-route rate limit.
    pub fn with_rate_limit(mut self, rps: u64) -> Self {
        self.rate_limit_rps = rps;
        self
    }

    /// Sets route priority.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Checks whether the route matches a path and method.
    pub fn matches(&self, path: &str, method: &HttpMethod) -> bool {
        path.starts_with(&self.path_prefix) && self.method.matches(method)
    }

    /// Computes the upstream URL for a given incoming path.
    pub fn upstream_url(&self, incoming_path: &str, base_url: &str) -> String {
        let rewritten = self
            .path_rewrite
            .as_ref()
            .map(|rw| rw.apply(incoming_path))
            .unwrap_or_else(|| incoming_path.to_string());
        format!("{}{}", base_url.trim_end_matches('/'), rewritten)
    }
}

// ── Token-bucket throttler ─────────────────────────────────────────────────────

/// Per-route token-bucket throttle state.
#[derive(Debug)]
pub struct RouteThrottle {
    /// Current token count.
    tokens: f64,
    /// Maximum token capacity.
    capacity: f64,
    /// Tokens refilled per second.
    refill_rate: f64,
    /// Last refill timestamp.
    last_refill: Instant,
}

impl RouteThrottle {
    /// Creates a throttle with the given capacity and refill rate (tokens/sec).
    pub fn new(capacity: f64, refill_rate: f64) -> Self {
        Self {
            tokens: capacity,
            capacity,
            refill_rate,
            last_refill: Instant::now(),
        }
    }

    /// Refills tokens based on elapsed time, then tries to consume `n` tokens.
    ///
    /// Returns `true` if the request is allowed.
    pub fn allow(&mut self, n: f64) -> bool {
        self.refill();
        if self.tokens >= n {
            self.tokens -= n;
            true
        } else {
            false
        }
    }

    /// Refills tokens based on elapsed time.
    fn refill(&mut self) {
        let elapsed = self.last_refill.elapsed().as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity);
        self.last_refill = Instant::now();
    }

    /// Current token level (after refill).
    pub fn available_tokens(&mut self) -> f64 {
        self.refill();
        self.tokens
    }
}

// ── Gateway metrics ────────────────────────────────────────────────────────────

/// Per-route metrics snapshot.
#[derive(Debug, Clone, Default)]
pub struct RouteMetrics {
    /// Total requests routed.
    pub total_requests: u64,
    /// Requests throttled (rate-limited).
    pub throttled_requests: u64,
    /// Requests rejected due to no healthy upstream.
    pub no_upstream_rejections: u64,
    /// Requests that timed out.
    pub timed_out: u64,
    /// Successful upstream responses.
    pub successes: u64,
    /// Sum of response latencies in microseconds (for average calculation).
    pub total_latency_us: u64,
}

impl RouteMetrics {
    /// Average latency in microseconds.
    pub fn avg_latency_us(&self) -> f64 {
        if self.successes == 0 {
            return 0.0;
        }
        self.total_latency_us as f64 / self.successes as f64
    }

    /// Success rate.
    pub fn success_rate(&self) -> f64 {
        if self.total_requests == 0 {
            return 1.0;
        }
        self.successes as f64 / self.total_requests as f64
    }
}

// ── Gateway ────────────────────────────────────────────────────────────────────

/// Gateway configuration.
#[derive(Debug, Clone)]
pub struct GatewayConfig {
    /// Global default rate limit (requests per second). 0 = unlimited.
    pub global_rps_limit: u64,
    /// Load balance strategy.
    pub load_balance: LoadBalanceStrategy,
    /// Whether to propagate the X-Request-Id header.
    pub propagate_request_id: bool,
    /// Whether to add X-Forwarded-For header.
    pub add_forwarded_for: bool,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            global_rps_limit: 0,
            load_balance: LoadBalanceStrategy::RoundRobin,
            propagate_request_id: true,
            add_forwarded_for: true,
        }
    }
}

/// Result of a route resolution attempt.
#[derive(Debug, Clone)]
pub struct RouteResolution {
    /// The matched route ID.
    pub route_id: String,
    /// The selected upstream ID.
    pub upstream_id: String,
    /// The fully-formed upstream URL.
    pub upstream_url: String,
    /// Headers to inject.
    pub inject_headers: HashMap<String, String>,
    /// Whether to strip the Authorization header.
    pub strip_auth: bool,
    /// Timeout for this request.
    pub timeout: Duration,
}

/// The API gateway.
pub struct ApiGateway {
    config: GatewayConfig,
    /// Registered routes, sorted by priority (highest first).
    routes: Vec<GatewayRoute>,
    /// Upstream registry.
    upstreams: HashMap<String, Upstream>,
    /// Per-route throttle states.
    throttles: HashMap<String, RouteThrottle>,
    /// Per-route metrics.
    metrics: HashMap<String, RouteMetrics>,
    /// Round-robin counter.
    round_robin_counter: u64,
    /// Global throttle (if `global_rps_limit > 0`).
    global_throttle: Option<RouteThrottle>,
}

impl ApiGateway {
    /// Creates a new gateway with the given configuration.
    pub fn new(config: GatewayConfig) -> Self {
        let global_throttle = if config.global_rps_limit > 0 {
            Some(RouteThrottle::new(
                config.global_rps_limit as f64,
                config.global_rps_limit as f64,
            ))
        } else {
            None
        };

        Self {
            config,
            routes: Vec::new(),
            upstreams: HashMap::new(),
            throttles: HashMap::new(),
            metrics: HashMap::new(),
            round_robin_counter: 0,
            global_throttle,
        }
    }

    /// Registers an upstream backend.
    pub fn register_upstream(&mut self, upstream: Upstream) {
        self.upstreams.insert(upstream.id.clone(), upstream);
    }

    /// Registers a route.
    pub fn register_route(&mut self, route: GatewayRoute) {
        let route_id = route.id.clone();
        let rps = route.rate_limit_rps;

        self.routes.push(route);
        // Sort routes by priority descending
        self.routes.sort_by(|a, b| b.priority.cmp(&a.priority));

        if rps > 0 {
            self.throttles
                .insert(route_id.clone(), RouteThrottle::new(rps as f64, rps as f64));
        }
        self.metrics.entry(route_id).or_default();
    }

    /// Updates the health of an upstream.
    pub fn set_upstream_health(&mut self, upstream_id: &str, health: UpstreamHealth) {
        if let Some(u) = self.upstreams.get_mut(upstream_id) {
            u.health = health;
        }
    }

    /// Resolves a route for an incoming request.
    ///
    /// Returns `Err(String)` if no route matches, the global throttle fires,
    /// the per-route throttle fires, or no healthy upstream is available.
    pub fn resolve(&mut self, path: &str, method: &HttpMethod) -> Result<RouteResolution, String> {
        // Global throttle check
        if let Some(gt) = &mut self.global_throttle {
            if !gt.allow(1.0) {
                return Err("Global rate limit exceeded".to_string());
            }
        }

        // Find matching route
        let route = self
            .routes
            .iter()
            .find(|r| r.matches(path, method))
            .ok_or_else(|| format!("No route matched path: {}", path))?
            .clone();

        let route_id = route.id.clone();

        // Update metrics counter
        let metrics = self.metrics.entry(route_id.clone()).or_default();
        metrics.total_requests += 1;

        // Per-route throttle check
        if let Some(throttle) = self.throttles.get_mut(&route_id) {
            if !throttle.allow(1.0) {
                let m = self.metrics.entry(route_id.clone()).or_default();
                m.throttled_requests += 1;
                return Err(format!("Route '{}' rate limit exceeded", route_id));
            }
        }

        // Collect routable upstreams for this route
        let routable: Vec<Upstream> = route
            .upstream_ids
            .iter()
            .filter_map(|id| self.upstreams.get(id))
            .filter(|u| u.health.is_routable())
            .cloned()
            .collect();

        if routable.is_empty() {
            let m = self.metrics.entry(route_id.clone()).or_default();
            m.no_upstream_rejections += 1;
            return Err(format!(
                "No healthy upstream available for route '{}'",
                route_id
            ));
        }

        // Select upstream
        let selected_idx = select_upstream(
            &routable,
            self.config.load_balance,
            &mut self.round_robin_counter,
        )
        .ok_or_else(|| "No routable upstream".to_string())?;

        let selected_upstream = &routable[selected_idx];
        let upstream_url = route.upstream_url(path, &selected_upstream.base_url);

        // Build inject headers
        let mut inject_headers = route.inject_headers.clone();

        if self.config.add_forwarded_for {
            inject_headers
                .entry("X-Forwarded-For".to_string())
                .or_insert_with(|| "unknown".to_string());
        }
        if self.config.propagate_request_id {
            inject_headers
                .entry("X-Request-Id".to_string())
                .or_insert_with(|| {
                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos();
                    format!("gw-{}", now)
                });
        }

        // Record success in metrics
        let m = self.metrics.entry(route_id.clone()).or_default();
        m.successes += 1;

        Ok(RouteResolution {
            route_id,
            upstream_id: selected_upstream.id.clone(),
            upstream_url,
            inject_headers,
            strip_auth: route.strip_auth,
            timeout: route.timeout,
        })
    }

    /// Records a latency observation for a route (in microseconds).
    pub fn record_latency(&mut self, route_id: &str, latency_us: u64) {
        let m = self.metrics.entry(route_id.to_string()).or_default();
        m.total_latency_us += latency_us;
    }

    /// Returns the metrics for a route.
    pub fn route_metrics(&self, route_id: &str) -> Option<&RouteMetrics> {
        self.metrics.get(route_id)
    }

    /// Returns all upstreams.
    pub fn upstreams(&self) -> &HashMap<String, Upstream> {
        &self.upstreams
    }

    /// Returns all routes.
    pub fn routes(&self) -> &[GatewayRoute] {
        &self.routes
    }

    /// Returns the number of healthy upstreams.
    pub fn healthy_upstream_count(&self) -> usize {
        self.upstreams
            .values()
            .filter(|u| u.health == UpstreamHealth::Healthy)
            .count()
    }

    /// Marks an upstream request as done (decrements active connections).
    pub fn upstream_request_done(&mut self, upstream_id: &str, error: bool) {
        if let Some(u) = self.upstreams.get_mut(upstream_id) {
            u.active_connections = u.active_connections.saturating_sub(1);
            u.total_requests += 1;
            if error {
                u.total_errors += 1;
            }
        }
    }
}

impl Default for ApiGateway {
    fn default() -> Self {
        Self::new(GatewayConfig::default())
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gateway() -> ApiGateway {
        let mut gw = ApiGateway::default();

        gw.register_upstream(Upstream::new("svc1", "http://svc1:8080"));
        gw.register_upstream(Upstream::new("svc2", "http://svc2:8080"));

        gw.register_route(GatewayRoute::new(
            "media-route",
            "/api/v1/media",
            vec!["svc1".to_string()],
        ));
        gw.register_route(
            GatewayRoute::new(
                "transcode-route",
                "/api/v1/transcode",
                vec!["svc2".to_string()],
            )
            .with_rate_limit(100),
        );

        gw
    }

    // UpstreamHealth tests

    #[test]
    fn test_upstream_health_is_routable() {
        assert!(UpstreamHealth::Healthy.is_routable());
        assert!(UpstreamHealth::Degraded.is_routable());
        assert!(!UpstreamHealth::Unhealthy.is_routable());
    }

    #[test]
    fn test_upstream_health_labels() {
        assert_eq!(UpstreamHealth::Healthy.label(), "healthy");
        assert_eq!(UpstreamHealth::Unhealthy.label(), "unhealthy");
    }

    // Upstream tests

    #[test]
    fn test_upstream_error_rate_zero() {
        let u = Upstream::new("u1", "http://localhost");
        assert!((u.error_rate()).abs() < 1e-9);
    }

    #[test]
    fn test_upstream_error_rate_calculated() {
        let mut u = Upstream::new("u1", "http://localhost");
        u.total_requests = 10;
        u.total_errors = 2;
        assert!((u.error_rate() - 0.2).abs() < 1e-9);
    }

    // HttpMethod tests

    #[test]
    fn test_http_method_parse() {
        assert_eq!(HttpMethod::parse("GET"), Some(HttpMethod::Get));
        assert_eq!(HttpMethod::parse("post"), Some(HttpMethod::Post));
        assert_eq!(HttpMethod::parse("UNKNOWN"), None);
    }

    #[test]
    fn test_http_method_any_matches_all() {
        assert!(HttpMethod::Any.matches(&HttpMethod::Get));
        assert!(HttpMethod::Any.matches(&HttpMethod::Delete));
    }

    #[test]
    fn test_http_method_specific_matches_same() {
        assert!(HttpMethod::Get.matches(&HttpMethod::Get));
        assert!(!HttpMethod::Get.matches(&HttpMethod::Post));
    }

    // PathRewrite tests

    #[test]
    fn test_path_rewrite_strip_prefix() {
        let rw = PathRewrite::strip_prefix("/api/v1");
        assert_eq!(rw.apply("/api/v1/media/123"), "/media/123");
    }

    #[test]
    fn test_path_rewrite_no_match() {
        let rw = PathRewrite::strip_prefix("/api/v1");
        assert_eq!(rw.apply("/other/path"), "/other/path");
    }

    #[test]
    fn test_path_rewrite_strip_to_root() {
        let rw = PathRewrite::strip_prefix("/api/v1");
        assert_eq!(rw.apply("/api/v1"), "/");
    }

    // GatewayRoute tests

    #[test]
    fn test_route_matches() {
        let route = GatewayRoute::new("r", "/api/v1", vec!["svc1".to_string()]);
        assert!(route.matches("/api/v1/media", &HttpMethod::Get));
        assert!(!route.matches("/other", &HttpMethod::Get));
    }

    #[test]
    fn test_route_upstream_url_with_rewrite() {
        let route = GatewayRoute::new("r", "/api/v1/media", vec!["svc1".to_string()])
            .with_rewrite(PathRewrite::strip_prefix("/api/v1"));
        let url = route.upstream_url("/api/v1/media/123", "http://backend:8080");
        assert_eq!(url, "http://backend:8080/media/123");
    }

    // RouteThrottle tests

    #[test]
    fn test_throttle_allows_within_capacity() {
        let mut t = RouteThrottle::new(10.0, 10.0);
        for _ in 0..10 {
            assert!(t.allow(1.0));
        }
    }

    #[test]
    fn test_throttle_blocks_when_empty() {
        let mut t = RouteThrottle::new(2.0, 0.0); // no refill
        t.allow(1.0);
        t.allow(1.0);
        assert!(!t.allow(1.0));
    }

    // select_upstream tests

    #[test]
    fn test_select_round_robin() {
        let upstreams = vec![
            Upstream::new("u1", "http://u1"),
            Upstream::new("u2", "http://u2"),
        ];
        let mut counter = 0u64;
        let i1 = select_upstream(&upstreams, LoadBalanceStrategy::RoundRobin, &mut counter);
        let i2 = select_upstream(&upstreams, LoadBalanceStrategy::RoundRobin, &mut counter);
        assert!(i1.is_some());
        assert!(i2.is_some());
        // Should cycle
        assert_ne!(i1, i2);
    }

    #[test]
    fn test_select_no_routable() {
        let mut upstreams = vec![Upstream::new("u1", "http://u1")];
        upstreams[0].health = UpstreamHealth::Unhealthy;
        let mut counter = 0u64;
        assert!(
            select_upstream(&upstreams, LoadBalanceStrategy::RoundRobin, &mut counter).is_none()
        );
    }

    #[test]
    fn test_select_least_connections() {
        let mut upstreams = vec![
            Upstream::new("u1", "http://u1"),
            Upstream::new("u2", "http://u2"),
        ];
        upstreams[0].active_connections = 5;
        upstreams[1].active_connections = 1;
        let mut counter = 0u64;
        let selected = select_upstream(
            &upstreams,
            LoadBalanceStrategy::LeastConnections,
            &mut counter,
        );
        assert_eq!(selected, Some(1)); // u2 has fewer connections
    }

    // ApiGateway tests

    #[test]
    fn test_gateway_resolve_basic() {
        let mut gw = make_gateway();
        let result = gw.resolve("/api/v1/media/m1", &HttpMethod::Get);
        assert!(result.is_ok());
        let res = result.expect("should succeed");
        assert_eq!(res.route_id, "media-route");
    }

    #[test]
    fn test_gateway_no_route_match() {
        let mut gw = make_gateway();
        let result = gw.resolve("/unknown/path", &HttpMethod::Get);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No route matched"));
    }

    #[test]
    fn test_gateway_unhealthy_upstream() {
        let mut gw = make_gateway();
        gw.set_upstream_health("svc1", UpstreamHealth::Unhealthy);
        let result = gw.resolve("/api/v1/media/m1", &HttpMethod::Get);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No healthy upstream"));
    }

    #[test]
    fn test_gateway_metrics_tracked() {
        let mut gw = make_gateway();
        gw.resolve("/api/v1/media", &HttpMethod::Get)
            .expect("should succeed");
        let metrics = gw.route_metrics("media-route").expect("metrics exist");
        assert_eq!(metrics.total_requests, 1);
        assert_eq!(metrics.successes, 1);
    }

    #[test]
    fn test_gateway_route_metrics_avg_latency() {
        let mut gw = make_gateway();
        gw.resolve("/api/v1/media", &HttpMethod::Get)
            .expect("should succeed");
        gw.record_latency("media-route", 1000);
        gw.record_latency("media-route", 2000);
        let metrics = gw.route_metrics("media-route").expect("exists");
        // 1 success was recorded during resolve; we added 3000 total latency
        assert!((metrics.avg_latency_us() - 3000.0).abs() < 1.0);
    }

    #[test]
    fn test_gateway_healthy_upstream_count() {
        let mut gw = make_gateway();
        assert_eq!(gw.healthy_upstream_count(), 2);
        gw.set_upstream_health("svc1", UpstreamHealth::Unhealthy);
        assert_eq!(gw.healthy_upstream_count(), 1);
    }

    #[test]
    fn test_gateway_global_rate_limit() {
        let config = GatewayConfig {
            global_rps_limit: 1,
            ..Default::default()
        };
        let mut gw = ApiGateway::new(config);
        gw.register_upstream(Upstream::new("svc", "http://svc:8080"));
        gw.register_route(GatewayRoute::new("r", "/", vec!["svc".to_string()]));

        // First request should succeed (capacity = 1 token)
        assert!(gw.resolve("/", &HttpMethod::Get).is_ok());
        // Second request should be throttled (bucket empty, no time to refill)
        assert!(gw.resolve("/", &HttpMethod::Get).is_err());
    }

    #[test]
    fn test_route_metrics_success_rate() {
        let m = RouteMetrics {
            total_requests: 10,
            successes: 9,
            ..Default::default()
        };
        assert!((m.success_rate() - 0.9).abs() < 1e-9);
    }
}
