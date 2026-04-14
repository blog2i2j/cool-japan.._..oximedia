//! REST API endpoint definitions for external tool integration.
//!
//! This module defines the request/response types and routing table for a
//! lightweight review API that can be integrated with Slack, Jira, and
//! Shotgrid.  The actual HTTP server transport is outside the scope of this
//! crate; only the data types and handler logic are defined here.

use serde::{Deserialize, Serialize};

// ─── API request / response types ────────────────────────────────────────────

/// HTTP method used for an API route.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HttpMethod {
    /// HTTP GET.
    Get,
    /// HTTP POST.
    Post,
    /// HTTP PUT.
    Put,
    /// HTTP DELETE.
    Delete,
    /// HTTP PATCH.
    Patch,
}

/// A single API route definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiRoute {
    /// HTTP method.
    pub method: HttpMethod,
    /// URL path pattern (e.g. `/reviews/{id}`).
    pub path: String,
    /// Human-readable description of what this endpoint does.
    pub description: String,
}

/// Generic API response envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse<T: Serialize> {
    /// Whether the request succeeded.
    pub success: bool,
    /// Optional payload.
    pub data: Option<T>,
    /// Human-readable error message when `success` is `false`.
    pub error: Option<String>,
}

impl<T: Serialize> ApiResponse<T> {
    /// Create a successful response carrying `data`.
    #[must_use]
    pub fn ok(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
        }
    }

    /// Create an error response.
    #[must_use]
    pub fn err(message: impl Into<String>) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(message.into()),
        }
    }
}

// ─── Webhook payload types ──────────────────────────────────────────────────

/// The integration platform that sent a webhook event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntegrationPlatform {
    /// Slack interactive message or slash-command callback.
    Slack,
    /// Jira issue-change webhook.
    Jira,
    /// ShotGrid (formerly Shotgun) event webhook.
    ShotGrid,
}

impl std::fmt::Display for IntegrationPlatform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Slack => write!(f, "slack"),
            Self::Jira => write!(f, "jira"),
            Self::ShotGrid => write!(f, "shotgrid"),
        }
    }
}

/// A webhook event received from an external integration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookEvent {
    /// Which platform sent the event.
    pub platform: IntegrationPlatform,
    /// The event type as reported by the platform (e.g. `"issue_updated"`).
    pub event_type: String,
    /// Opaque JSON payload from the platform (stored as a string to avoid
    /// pulling in `serde_json::Value` at this layer).
    pub payload: String,
    /// Timestamp the event was received (ms since UNIX epoch).
    pub received_at_ms: u64,
}

/// Configuration for connecting a review session to an external integration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// The platform this configuration targets.
    pub platform: IntegrationPlatform,
    /// Webhook URL to push notifications to.
    pub webhook_url: String,
    /// Optional authentication token/secret for the webhook.
    pub auth_token: Option<String>,
    /// Whether this integration is currently enabled.
    pub enabled: bool,
}

impl IntegrationConfig {
    /// Create a new integration config.
    #[must_use]
    pub fn new(platform: IntegrationPlatform, webhook_url: impl Into<String>) -> Self {
        Self {
            platform,
            webhook_url: webhook_url.into(),
            auth_token: None,
            enabled: true,
        }
    }

    /// Set the auth token (builder pattern).
    #[must_use]
    pub fn with_auth_token(mut self, token: impl Into<String>) -> Self {
        self.auth_token = Some(token.into());
        self
    }
}

/// Matches a concrete URL path against a pattern containing `{param}` placeholders.
///
/// Returns `Some(params)` where `params` maps placeholder names to their matched
/// segments, or `None` if the path does not match.
#[must_use]
pub fn match_route_path(
    pattern: &str,
    path: &str,
) -> Option<std::collections::HashMap<String, String>> {
    let pat_segments: Vec<&str> = pattern.split('/').collect();
    let path_segments: Vec<&str> = path.split('/').collect();

    if pat_segments.len() != path_segments.len() {
        return None;
    }

    let mut params = std::collections::HashMap::new();
    for (pat, seg) in pat_segments.iter().zip(path_segments.iter()) {
        if pat.starts_with('{') && pat.ends_with('}') {
            let name = &pat[1..pat.len() - 1];
            params.insert(name.to_string(), (*seg).to_string());
        } else if pat != seg {
            return None;
        }
    }
    Some(params)
}

/// Find the first route that matches a given method and path.
///
/// Returns the matched `ApiRoute` and any extracted path parameters.
#[must_use]
pub fn resolve_route(
    method: HttpMethod,
    path: &str,
) -> Option<(ApiRoute, std::collections::HashMap<String, String>)> {
    for route in ReviewApiRoutes::all() {
        if route.method == method {
            if let Some(params) = match_route_path(&route.path, path) {
                return Some((route, params));
            }
        }
    }
    None
}

// ─── Review API route table ───────────────────────────────────────────────────

/// The complete set of REST routes exposed by the review API.
pub struct ReviewApiRoutes;

impl ReviewApiRoutes {
    /// Return all defined API routes.
    #[must_use]
    pub fn all() -> Vec<ApiRoute> {
        vec![
            ApiRoute {
                method: HttpMethod::Get,
                path: "/reviews".to_string(),
                description: "List all review sessions".to_string(),
            },
            ApiRoute {
                method: HttpMethod::Post,
                path: "/reviews".to_string(),
                description: "Create a new review session".to_string(),
            },
            ApiRoute {
                method: HttpMethod::Get,
                path: "/reviews/{id}".to_string(),
                description: "Get a specific review session".to_string(),
            },
            ApiRoute {
                method: HttpMethod::Put,
                path: "/reviews/{id}".to_string(),
                description: "Update a review session".to_string(),
            },
            ApiRoute {
                method: HttpMethod::Delete,
                path: "/reviews/{id}".to_string(),
                description: "Delete a review session".to_string(),
            },
            ApiRoute {
                method: HttpMethod::Get,
                path: "/reviews/{id}/comments".to_string(),
                description: "List comments for a review".to_string(),
            },
            ApiRoute {
                method: HttpMethod::Post,
                path: "/reviews/{id}/comments".to_string(),
                description: "Add a comment to a review".to_string(),
            },
            ApiRoute {
                method: HttpMethod::Post,
                path: "/reviews/{id}/approve".to_string(),
                description: "Approve a review".to_string(),
            },
            ApiRoute {
                method: HttpMethod::Post,
                path: "/reviews/{id}/reject".to_string(),
                description: "Reject a review".to_string(),
            },
            ApiRoute {
                method: HttpMethod::Get,
                path: "/reviews/{id}/export".to_string(),
                description: "Export review data (JSON, CSV, PDF)".to_string(),
            },
            // Webhook / integration endpoints
            ApiRoute {
                method: HttpMethod::Post,
                path: "/webhooks/slack".to_string(),
                description: "Receive Slack action callbacks".to_string(),
            },
            ApiRoute {
                method: HttpMethod::Post,
                path: "/webhooks/jira".to_string(),
                description: "Receive Jira issue update webhooks".to_string(),
            },
            ApiRoute {
                method: HttpMethod::Post,
                path: "/webhooks/shotgrid".to_string(),
                description: "Receive ShotGrid event webhooks".to_string(),
            },
        ]
    }

    /// Find routes matching the given HTTP method.
    #[must_use]
    pub fn by_method(method: HttpMethod) -> Vec<ApiRoute> {
        Self::all()
            .into_iter()
            .filter(|r| r.method == method)
            .collect()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_non_empty() {
        assert!(!ReviewApiRoutes::all().is_empty());
    }

    #[test]
    fn routes_include_get_and_post() {
        let all = ReviewApiRoutes::all();
        assert!(all.iter().any(|r| r.method == HttpMethod::Get));
        assert!(all.iter().any(|r| r.method == HttpMethod::Post));
    }

    #[test]
    fn api_response_ok_has_data() {
        let resp: ApiResponse<i32> = ApiResponse::ok(42);
        assert!(resp.success);
        assert_eq!(resp.data, Some(42));
        assert!(resp.error.is_none());
    }

    #[test]
    fn api_response_err_has_message() {
        let resp: ApiResponse<i32> = ApiResponse::err("not found");
        assert!(!resp.success);
        assert!(resp.data.is_none());
        assert_eq!(resp.error.as_deref(), Some("not found"));
    }

    #[test]
    fn by_method_get_returns_get_routes() {
        let gets = ReviewApiRoutes::by_method(HttpMethod::Get);
        assert!(!gets.is_empty());
        for r in &gets {
            assert_eq!(r.method, HttpMethod::Get);
        }
    }

    #[test]
    fn by_method_delete_returns_delete_routes() {
        let deletes = ReviewApiRoutes::by_method(HttpMethod::Delete);
        assert!(!deletes.is_empty());
        for r in &deletes {
            assert_eq!(r.method, HttpMethod::Delete);
        }
    }

    #[test]
    fn routes_include_webhook_paths() {
        let all = ReviewApiRoutes::all();
        assert!(all.iter().any(|r| r.path.contains("/webhooks/slack")));
        assert!(all.iter().any(|r| r.path.contains("/webhooks/jira")));
        assert!(all.iter().any(|r| r.path.contains("/webhooks/shotgrid")));
    }

    #[test]
    fn match_route_path_exact() {
        let params = match_route_path("/reviews", "/reviews");
        assert!(params.is_some());
        assert!(params.as_ref().map_or(false, |p| p.is_empty()));
    }

    #[test]
    fn match_route_path_with_param() {
        let params = match_route_path("/reviews/{id}", "/reviews/abc-123");
        assert!(params.is_some());
        let params = params.expect("should match");
        assert_eq!(params.get("id").map(|s| s.as_str()), Some("abc-123"));
    }

    #[test]
    fn match_route_path_no_match() {
        let params = match_route_path("/reviews/{id}", "/users/1");
        assert!(params.is_none());
    }

    #[test]
    fn resolve_route_finds_get_reviews() {
        let result = resolve_route(HttpMethod::Get, "/reviews");
        assert!(result.is_some());
        let (route, _params) = result.expect("should resolve");
        assert_eq!(route.method, HttpMethod::Get);
        assert_eq!(route.path, "/reviews");
    }

    #[test]
    fn resolve_route_extracts_id_param() {
        let result = resolve_route(HttpMethod::Get, "/reviews/my-session");
        assert!(result.is_some());
        let (_route, params) = result.expect("should resolve");
        assert_eq!(params.get("id").map(|s| s.as_str()), Some("my-session"));
    }

    #[test]
    fn resolve_route_returns_none_for_unknown() {
        let result = resolve_route(HttpMethod::Get, "/nonexistent/path");
        assert!(result.is_none());
    }

    #[test]
    fn integration_config_builder() {
        let cfg = IntegrationConfig::new(IntegrationPlatform::Slack, "https://hooks.slack.com/xxx")
            .with_auth_token("tok-123");
        assert_eq!(cfg.platform, IntegrationPlatform::Slack);
        assert!(cfg.enabled);
        assert_eq!(cfg.auth_token.as_deref(), Some("tok-123"));
    }

    #[test]
    fn webhook_event_serialization() {
        let evt = WebhookEvent {
            platform: IntegrationPlatform::Jira,
            event_type: "issue_updated".to_string(),
            payload: r#"{"key":"PROJ-1"}"#.to_string(),
            received_at_ms: 1_700_000_000_000,
        };
        assert_eq!(evt.platform, IntegrationPlatform::Jira);
        assert_eq!(evt.event_type, "issue_updated");
    }

    #[test]
    fn integration_platform_display() {
        assert_eq!(IntegrationPlatform::Slack.to_string(), "slack");
        assert_eq!(IntegrationPlatform::Jira.to_string(), "jira");
        assert_eq!(IntegrationPlatform::ShotGrid.to_string(), "shotgrid");
    }
}
