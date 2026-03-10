//! CDN configuration and management abstractions.
//!
//! This module provides types for modelling CDN providers, origins, cache
//! rules, configurations, and aggregated metrics — without making any real
//! network calls.

#![allow(dead_code)]

/// Supported CDN providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CdnProviderKind {
    /// Amazon CloudFront.
    CloudFront,
    /// Fastly.
    Fastly,
    /// Cloudflare.
    Cloudflare,
    /// Akamai.
    Akamai,
    /// Microsoft Azure CDN.
    Azure,
    /// Google Cloud CDN.
    GcpCdn,
}

impl CdnProviderKind {
    /// Approximate number of Points of Presence (PoPs) for this provider.
    #[must_use]
    pub fn typical_pop_count(&self) -> u32 {
        match self {
            CdnProviderKind::CloudFront => 400,
            CdnProviderKind::Fastly => 70,
            CdnProviderKind::Cloudflare => 300,
            CdnProviderKind::Akamai => 4100,
            CdnProviderKind::Azure => 120,
            CdnProviderKind::GcpCdn => 100,
        }
    }

    /// Returns `true` when this CDN provider supports live-streaming workloads.
    #[must_use]
    pub fn supports_live(&self) -> bool {
        matches!(
            self,
            CdnProviderKind::CloudFront
                | CdnProviderKind::Fastly
                | CdnProviderKind::Cloudflare
                | CdnProviderKind::Akamai
        )
    }
}

/// Protocol used to pull content from a CDN origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CdnProtocol {
    /// Plain HTTP.
    Http,
    /// HTTPS.
    Https,
    /// RTMP live stream.
    Rtmp,
    /// HLS adaptive stream.
    Hls,
    /// MPEG-DASH adaptive stream.
    Dash,
}

impl CdnProtocol {
    /// Returns `true` when this protocol is a streaming (media) protocol.
    #[must_use]
    pub fn is_streaming(&self) -> bool {
        matches!(
            self,
            CdnProtocol::Rtmp | CdnProtocol::Hls | CdnProtocol::Dash
        )
    }
}

/// An origin server that a CDN pulls content from.
#[derive(Debug, Clone)]
pub struct CdnOrigin {
    /// URL of the origin.
    pub url: String,
    /// Protocol used to communicate with the origin.
    pub protocol: CdnProtocol,
    /// Optional failover origin URL.
    pub failover_url: Option<String>,
}

impl CdnOrigin {
    /// Creates a new `CdnOrigin`.
    #[must_use]
    pub fn new(
        url: impl Into<String>,
        protocol: CdnProtocol,
        failover_url: Option<String>,
    ) -> Self {
        Self {
            url: url.into(),
            protocol,
            failover_url,
        }
    }

    /// Returns `true` when a failover URL is configured.
    #[must_use]
    pub fn has_failover(&self) -> bool {
        self.failover_url.is_some()
    }
}

/// A cache rule that controls TTL and compression for paths matching a pattern.
#[derive(Debug, Clone)]
pub struct CacheRule {
    /// Path pattern (prefix or suffix) to match.
    pub path_pattern: String,
    /// Time-to-live in seconds for cached responses.
    pub ttl_seconds: u32,
    /// Whether responses should be compressed.
    pub compress: bool,
}

impl CacheRule {
    /// Creates a new `CacheRule`.
    #[must_use]
    pub fn new(path_pattern: impl Into<String>, ttl_seconds: u32, compress: bool) -> Self {
        Self {
            path_pattern: path_pattern.into(),
            ttl_seconds,
            compress,
        }
    }

    /// Returns `true` when `path` matches the rule's pattern.
    ///
    /// A pattern ending with `*` is treated as a prefix match; a pattern
    /// starting with `*` is treated as a suffix match; otherwise the pattern
    /// must appear as a substring of the path.
    #[must_use]
    pub fn matches(&self, path: &str) -> bool {
        let pat = self.path_pattern.as_str();
        if pat.ends_with('*') {
            path.starts_with(&pat[..pat.len() - 1])
        } else if pat.starts_with('*') {
            path.ends_with(&pat[1..])
        } else {
            path.contains(pat)
        }
    }
}

/// Full CDN configuration for a distribution.
#[derive(Debug, Clone)]
pub struct CdnConfig {
    /// CDN provider.
    pub provider: CdnProviderKind,
    /// Origin server.
    pub origin: CdnOrigin,
    /// Ordered list of cache rules.
    pub cache_rules: Vec<CacheRule>,
    /// Optional custom domain name.
    pub custom_domain: Option<String>,
}

impl CdnConfig {
    /// Creates a new `CdnConfig`.
    #[must_use]
    pub fn new(
        provider: CdnProviderKind,
        origin: CdnOrigin,
        custom_domain: Option<String>,
    ) -> Self {
        Self {
            provider,
            origin,
            cache_rules: Vec::new(),
            custom_domain,
        }
    }

    /// Adds a cache rule.
    pub fn add_cache_rule(&mut self, rule: CacheRule) {
        self.cache_rules.push(rule);
    }

    /// Finds the first cache rule that matches `path`.
    #[must_use]
    pub fn find_rule(&self, path: &str) -> Option<&CacheRule> {
        self.cache_rules.iter().find(|r| r.matches(path))
    }

    /// Returns the TTL for `path` from the first matching rule, or 0 if none match.
    #[must_use]
    pub fn ttl_for(&self, path: &str) -> u32 {
        self.find_rule(path).map_or(0, |r| r.ttl_seconds)
    }
}

/// Aggregated CDN request / bandwidth metrics.
#[derive(Debug, Clone, Copy, Default)]
pub struct CdnMetricsSnapshot {
    /// Total HTTP requests served.
    pub requests: u64,
    /// Requests served from the edge cache (cache hits).
    pub cache_hits: u64,
    /// Total bytes served.
    pub bytes_served: u64,
    /// Number of error responses (4xx / 5xx).
    pub error_count: u64,
}

impl CdnMetricsSnapshot {
    /// Creates a new metrics snapshot.
    #[must_use]
    pub fn new(requests: u64, cache_hits: u64, bytes_served: u64, error_count: u64) -> Self {
        Self {
            requests,
            cache_hits,
            bytes_served,
            error_count,
        }
    }

    /// Cache hit rate (0.0 – 1.0).
    ///
    /// Returns 0.0 when `requests` is zero.
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        if self.requests == 0 {
            return 0.0;
        }
        self.cache_hits as f64 / self.requests as f64
    }

    /// Error rate (0.0 – 1.0).
    ///
    /// Returns 0.0 when `requests` is zero.
    #[must_use]
    pub fn error_rate(&self) -> f64 {
        if self.requests == 0 {
            return 0.0;
        }
        self.error_count as f64 / self.requests as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_origin() -> CdnOrigin {
        CdnOrigin::new("https://origin.example.com", CdnProtocol::Https, None)
    }

    // 1. CdnProviderKind::typical_pop_count
    #[test]
    fn test_pop_counts() {
        assert!(CdnProviderKind::CloudFront.typical_pop_count() > 0);
        assert!(
            CdnProviderKind::Akamai.typical_pop_count()
                > CdnProviderKind::Fastly.typical_pop_count()
        );
    }

    // 2. CdnProviderKind::supports_live
    #[test]
    fn test_supports_live() {
        assert!(CdnProviderKind::CloudFront.supports_live());
        assert!(CdnProviderKind::Fastly.supports_live());
        assert!(!CdnProviderKind::Azure.supports_live());
        assert!(!CdnProviderKind::GcpCdn.supports_live());
    }

    // 3. CdnProtocol::is_streaming
    #[test]
    fn test_protocol_is_streaming() {
        assert!(CdnProtocol::Rtmp.is_streaming());
        assert!(CdnProtocol::Hls.is_streaming());
        assert!(CdnProtocol::Dash.is_streaming());
        assert!(!CdnProtocol::Http.is_streaming());
        assert!(!CdnProtocol::Https.is_streaming());
    }

    // 4. CdnOrigin::has_failover
    #[test]
    fn test_origin_has_failover() {
        let with_failover = CdnOrigin::new(
            "https://a.example.com",
            CdnProtocol::Https,
            Some("https://b.example.com".to_string()),
        );
        assert!(with_failover.has_failover());
        assert!(!simple_origin().has_failover());
    }

    // 5. CacheRule::matches – prefix
    #[test]
    fn test_cache_rule_prefix_match() {
        let rule = CacheRule::new("/videos/*", 300, false);
        assert!(rule.matches("/videos/clip.mp4"));
        assert!(!rule.matches("/images/photo.jpg"));
    }

    // 6. CacheRule::matches – suffix
    #[test]
    fn test_cache_rule_suffix_match() {
        let rule = CacheRule::new("*.mp4", 600, false);
        assert!(rule.matches("/content/movie.mp4"));
        assert!(!rule.matches("/content/movie.webm"));
    }

    // 7. CacheRule::matches – substring
    #[test]
    fn test_cache_rule_substring_match() {
        let rule = CacheRule::new("/api/", 0, false);
        assert!(rule.matches("/api/status"));
        assert!(!rule.matches("/static/logo.png"));
    }

    // 8. CdnConfig::find_rule – first matching rule
    #[test]
    fn test_cdn_config_find_rule() {
        let mut cfg = CdnConfig::new(CdnProviderKind::CloudFront, simple_origin(), None);
        cfg.add_cache_rule(CacheRule::new("/videos/*", 300, false));
        cfg.add_cache_rule(CacheRule::new("/images/*", 3600, true));
        let rule = cfg
            .find_rule("/videos/trailer.mp4")
            .expect("rule should be valid");
        assert_eq!(rule.ttl_seconds, 300);
    }

    // 9. CdnConfig::find_rule – no match
    #[test]
    fn test_cdn_config_find_rule_none() {
        let cfg = CdnConfig::new(CdnProviderKind::Fastly, simple_origin(), None);
        assert!(cfg.find_rule("/unknown/path").is_none());
    }

    // 10. CdnConfig::ttl_for – matching rule
    #[test]
    fn test_cdn_config_ttl_for() {
        let mut cfg = CdnConfig::new(CdnProviderKind::Cloudflare, simple_origin(), None);
        cfg.add_cache_rule(CacheRule::new("/static/*", 86400, true));
        assert_eq!(cfg.ttl_for("/static/app.js"), 86400);
    }

    // 11. CdnConfig::ttl_for – no matching rule returns 0
    #[test]
    fn test_cdn_config_ttl_for_none() {
        let cfg = CdnConfig::new(CdnProviderKind::Azure, simple_origin(), None);
        assert_eq!(cfg.ttl_for("/dynamic/data"), 0);
    }

    // 12. CdnMetricsSnapshot::hit_rate
    #[test]
    fn test_metrics_hit_rate() {
        let m = CdnMetricsSnapshot::new(1000, 850, 10_000_000, 5);
        assert!((m.hit_rate() - 0.85).abs() < f64::EPSILON);
    }

    // 13. CdnMetricsSnapshot::error_rate
    #[test]
    fn test_metrics_error_rate() {
        let m = CdnMetricsSnapshot::new(200, 150, 0, 10);
        assert!((m.error_rate() - 0.05).abs() < f64::EPSILON);
    }

    // 14. CdnMetricsSnapshot – zero requests
    #[test]
    fn test_metrics_zero_requests() {
        let m = CdnMetricsSnapshot::new(0, 0, 0, 0);
        assert_eq!(m.hit_rate(), 0.0);
        assert_eq!(m.error_rate(), 0.0);
    }

    // 15. CdnConfig custom_domain
    #[test]
    fn test_cdn_config_custom_domain() {
        let cfg = CdnConfig::new(
            CdnProviderKind::GcpCdn,
            simple_origin(),
            Some("cdn.mysite.com".to_string()),
        );
        assert_eq!(cfg.custom_domain.as_deref(), Some("cdn.mysite.com"));
    }
}
