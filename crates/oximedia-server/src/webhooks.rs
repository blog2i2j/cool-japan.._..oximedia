//! Webhook delivery system for OxiMedia server events.
//!
//! Webhooks allow external services to receive real-time notifications when
//! significant events occur (transcode completed/failed, media upload/delete).
//! Each delivery is signed with HMAC-SHA256 so the receiver can verify origin.
//!
//! # Retry policy
//! Up to 3 delivery attempts with exponential backoff: 1 s → 4 s → 16 s.

use crate::error::{ServerError, ServerResult};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use hex::ToHex;
use hmac::{Hmac, KeyInit, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

// ── Domain types ──────────────────────────────────────────────────────────────

/// Events that can be emitted and delivered to registered webhooks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WebhookEvent {
    /// A transcode job finished successfully.
    TranscodeCompleted {
        /// Job identifier.
        job_id: String,
        /// Media item that was transcoded.
        media_id: String,
        /// URL of the output file.
        output_url: String,
    },
    /// A transcode job failed.
    TranscodeFailed {
        /// Job identifier.
        job_id: String,
        /// Media item that was transcoded.
        media_id: String,
        /// Error description.
        error: String,
    },
    /// A media file was successfully uploaded.
    MediaUploaded {
        /// New media ID.
        media_id: String,
        /// Original filename.
        filename: String,
        /// File size in bytes.
        size_bytes: u64,
    },
    /// A media item was deleted.
    MediaDeleted {
        /// Deleted media ID.
        media_id: String,
    },
}

impl WebhookEvent {
    /// Returns the event-type string used for filtering (e.g. "transcode.completed").
    pub fn event_type_name(&self) -> &'static str {
        match self {
            Self::TranscodeCompleted { .. } => "transcode.completed",
            Self::TranscodeFailed { .. } => "transcode.failed",
            Self::MediaUploaded { .. } => "media.uploaded",
            Self::MediaDeleted { .. } => "media.deleted",
        }
    }
}

/// Registration configuration for a single webhook endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookConfig {
    /// Stable webhook identifier.
    pub id: String,
    /// Target URL that receives POST requests.
    pub url: String,
    /// Event-type strings this webhook is subscribed to.
    pub events: Vec<String>,
    /// HMAC-SHA256 signing secret.  **Never returned in API responses.**
    #[serde(skip_serializing)]
    pub secret: String,
    /// Whether the webhook is active.
    pub active: bool,
}

/// Status of a single delivery attempt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryStatus {
    /// Delivery succeeded (HTTP 2xx).
    Success,
    /// Delivery failed and will not be retried further.
    Failed,
    /// Delivery is pending or in-flight.
    Pending,
}

/// Record of a webhook delivery attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookDelivery {
    /// Delivery identifier.
    pub id: String,
    /// Which webhook this delivery belongs to.
    pub webhook_id: String,
    /// Event-type name.
    pub event: String,
    /// Full JSON payload that was (or will be) sent.
    pub payload: serde_json::Value,
    /// Outcome of the delivery.
    pub status: DeliveryStatus,
    /// Number of attempts made so far.
    pub attempts: u32,
    /// Unix timestamp when the delivery record was created.
    pub created_at: i64,
}

// ── Request / response bodies ─────────────────────────────────────────────────

/// Request body for `POST /api/v1/webhooks`.
#[derive(Debug, Deserialize)]
pub struct RegisterWebhookRequest {
    /// Target URL.
    pub url: String,
    /// List of event-type strings to subscribe to.
    pub events: Vec<String>,
    /// Signing secret.  If omitted a random 32-byte hex secret is generated.
    pub secret: Option<String>,
}

/// Request body for `PUT /api/v1/webhooks/{id}`.
#[derive(Debug, Deserialize)]
pub struct UpdateWebhookRequest {
    /// Updated target URL (optional).
    pub url: Option<String>,
    /// Updated event subscriptions (optional).
    pub events: Option<Vec<String>>,
    /// New signing secret (optional).
    pub secret: Option<String>,
    /// Enable/disable the webhook (optional).
    pub active: Option<bool>,
}

// ── WebhookManager ────────────────────────────────────────────────────────────

/// In-memory registry and delivery engine for webhooks.
///
/// In a production deployment this would be backed by the database; here we use
/// an in-process `RwLock<Vec<...>>` to keep the implementation free of compile-time
/// SQLx queries while still being fully functional and testable.
pub struct WebhookManager {
    configs: RwLock<Vec<WebhookConfig>>,
    deliveries: RwLock<Vec<WebhookDelivery>>,
    /// Optional HTTP client — `None` in unit-test environments where no TLS
    /// provider is available.  Constructed lazily on first delivery attempt.
    http: Option<reqwest::Client>,
}

impl WebhookManager {
    /// Creates a new `WebhookManager`, attempting to build an HTTP client.
    ///
    /// If no TLS provider is installed (e.g. in unit tests), the HTTP client
    /// will be absent and delivery attempts will be skipped with a warning.
    #[must_use]
    pub fn new() -> Self {
        // Use catch_unwind to guard against reqwest panicking when no TLS
        // provider is installed (happens in test environments that don't call
        // `rustls::crypto::ring::default_provider().install_default()`).
        let http = std::panic::catch_unwind(|| {
            reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .ok()
        })
        .ok()
        .flatten();

        Self {
            configs: RwLock::new(Vec::new()),
            deliveries: RwLock::new(Vec::new()),
            http,
        }
    }

    /// Creates a `WebhookManager` without an HTTP client (for testing).
    #[cfg(test)]
    #[must_use]
    pub fn new_for_test() -> Self {
        Self {
            configs: RwLock::new(Vec::new()),
            deliveries: RwLock::new(Vec::new()),
            http: None,
        }
    }

    /// Registers a new webhook and returns its configuration.
    ///
    /// # Errors
    ///
    /// Returns `ServerError::BadRequest` when the URL is empty or the event
    /// list is empty.
    pub async fn register(&self, req: RegisterWebhookRequest) -> ServerResult<WebhookConfig> {
        if req.url.trim().is_empty() {
            return Err(ServerError::BadRequest(
                "webhook url must not be empty".to_string(),
            ));
        }
        if req.events.is_empty() {
            return Err(ServerError::BadRequest(
                "webhook must subscribe to at least one event".to_string(),
            ));
        }

        let secret = req
            .secret
            .filter(|s| !s.is_empty())
            .unwrap_or_else(generate_secret);

        let config = WebhookConfig {
            id: Uuid::new_v4().to_string(),
            url: req.url,
            events: req.events,
            secret,
            active: true,
        };

        self.configs.write().await.push(config.clone());
        Ok(config)
    }

    /// Returns all registered webhooks (secrets stripped).
    pub async fn list(&self) -> Vec<WebhookConfig> {
        self.configs.read().await.clone()
    }

    /// Returns a single webhook by ID (secret stripped).
    ///
    /// # Errors
    ///
    /// Returns `ServerError::NotFound` when the ID is unknown.
    pub async fn get(&self, id: &str) -> ServerResult<WebhookConfig> {
        self.configs
            .read()
            .await
            .iter()
            .find(|c| c.id == id)
            .cloned()
            .ok_or_else(|| ServerError::NotFound(format!("Webhook '{}' not found", id)))
    }

    /// Updates an existing webhook configuration.
    ///
    /// # Errors
    ///
    /// Returns `ServerError::NotFound` when the ID is unknown.
    pub async fn update(&self, id: &str, req: UpdateWebhookRequest) -> ServerResult<WebhookConfig> {
        let mut configs = self.configs.write().await;
        let cfg = configs
            .iter_mut()
            .find(|c| c.id == id)
            .ok_or_else(|| ServerError::NotFound(format!("Webhook '{}' not found", id)))?;

        if let Some(url) = req.url {
            cfg.url = url;
        }
        if let Some(events) = req.events {
            cfg.events = events;
        }
        if let Some(secret) = req.secret {
            cfg.secret = secret;
        }
        if let Some(active) = req.active {
            cfg.active = active;
        }

        Ok(cfg.clone())
    }

    /// Removes a webhook registration.
    ///
    /// # Errors
    ///
    /// Returns `ServerError::NotFound` when the ID is unknown.
    pub async fn delete(&self, id: &str) -> ServerResult<()> {
        let mut configs = self.configs.write().await;
        let pos = configs
            .iter()
            .position(|c| c.id == id)
            .ok_or_else(|| ServerError::NotFound(format!("Webhook '{}' not found", id)))?;
        configs.remove(pos);
        Ok(())
    }

    /// Returns all deliveries for a given webhook ID.
    pub async fn deliveries_for(&self, webhook_id: &str) -> Vec<WebhookDelivery> {
        self.deliveries
            .read()
            .await
            .iter()
            .filter(|d| d.webhook_id == webhook_id)
            .cloned()
            .collect()
    }

    /// Fans out `event` to all active, subscribed webhooks.
    ///
    /// Non-blocking: spawns a background task per matching webhook.
    /// Each task retries up to 3 times with exponential backoff (1 s, 4 s, 16 s).
    pub async fn deliver(&self, event: &WebhookEvent) {
        let event_name = event.event_type_name();

        // Collect matching configs — hold the lock only briefly.
        let matching: Vec<WebhookConfig> = self
            .configs
            .read()
            .await
            .iter()
            .filter(|c| c.active && c.events.iter().any(|e| e == event_name))
            .cloned()
            .collect();

        let payload = match serde_json::to_value(event) {
            Ok(v) => v,
            Err(e) => {
                tracing::error!("webhook: failed to serialize event: {}", e);
                return;
            }
        };

        // If no HTTP client is available (e.g. unit tests without TLS provider),
        // log a warning and skip actual delivery.
        let http = match &self.http {
            Some(c) => c.clone(),
            None => {
                tracing::warn!(
                    "webhook: HTTP client unavailable; skipping delivery of '{}'",
                    event_name
                );
                return;
            }
        };

        for cfg in matching {
            let payload_clone = payload.clone();
            let http = http.clone();

            // Record a pending delivery.
            let delivery_id = Uuid::new_v4().to_string();
            let delivery = WebhookDelivery {
                id: delivery_id.clone(),
                webhook_id: cfg.id.clone(),
                event: event_name.to_string(),
                payload: payload_clone.clone(),
                status: DeliveryStatus::Pending,
                attempts: 0,
                created_at: chrono::Utc::now().timestamp(),
            };
            self.deliveries.write().await.push(delivery);

            tokio::spawn(async move {
                attempt_delivery(http, cfg, payload_clone, delivery_id).await;
            });
        }
    }
}

impl Default for WebhookManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Delivery internals ────────────────────────────────────────────────────────

/// Exponential backoff delays for retry attempts (seconds).
const RETRY_DELAYS_SECS: [u64; 3] = [1, 4, 16];

/// Attempts to deliver `payload` to the configured URL, retrying on failure.
async fn attempt_delivery(
    http: reqwest::Client,
    cfg: WebhookConfig,
    payload: serde_json::Value,
    delivery_id: String,
) {
    let body = match serde_json::to_string(&payload) {
        Ok(b) => b,
        Err(e) => {
            tracing::error!("webhook delivery {}: serialize error: {}", delivery_id, e);
            return;
        }
    };
    let signature = sign_payload(&cfg.secret, &body);

    for (attempt, &delay_secs) in RETRY_DELAYS_SECS.iter().enumerate() {
        if attempt > 0 {
            tokio::time::sleep(std::time::Duration::from_secs(delay_secs)).await;
        }

        match http
            .post(&cfg.url)
            .header("Content-Type", "application/json")
            .header("X-Oximedia-Signature", format!("sha256={}", signature))
            .header("X-Oximedia-Delivery", &delivery_id)
            .body(body.clone())
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                tracing::info!(
                    "webhook delivery {} to {} succeeded (attempt {})",
                    delivery_id,
                    cfg.url,
                    attempt + 1
                );
                return;
            }
            Ok(resp) => {
                tracing::warn!(
                    "webhook delivery {} to {} HTTP {} (attempt {})",
                    delivery_id,
                    cfg.url,
                    resp.status(),
                    attempt + 1,
                );
            }
            Err(e) => {
                tracing::warn!(
                    "webhook delivery {} to {} error: {} (attempt {})",
                    delivery_id,
                    cfg.url,
                    e,
                    attempt + 1,
                );
            }
        }
    }

    tracing::error!(
        "webhook delivery {} to {} exhausted all retries",
        delivery_id,
        cfg.url
    );
}

/// Computes `HMAC-SHA256(secret, body)` and returns the lower-hex digest.
fn sign_payload(secret: &str, body: &str) -> String {
    let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes()).unwrap_or_else(|_| {
        // new_from_slice only fails for zero-length keys; use a fallback key.
        Hmac::<Sha256>::new_from_slice(b"__fallback__").expect("fallback key always valid")
    });
    mac.update(body.as_bytes());
    mac.finalize().into_bytes().encode_hex::<String>()
}

/// Generates a cryptographically random 32-byte hex secret.
fn generate_secret() -> String {
    use rand::Rng;
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    bytes.encode_hex::<String>()
}

// ── Route handlers ────────────────────────────────────────────────────────────

/// `GET /api/v1/webhooks` — list all registered webhooks.
pub async fn list_webhooks(State(manager): State<Arc<WebhookManager>>) -> impl IntoResponse {
    let list = manager.list().await;
    Json(list)
}

/// `POST /api/v1/webhooks` — register a new webhook.
pub async fn create_webhook(
    State(manager): State<Arc<WebhookManager>>,
    Json(body): Json<RegisterWebhookRequest>,
) -> Result<impl IntoResponse, crate::error::ServerError> {
    let cfg = manager.register(body).await?;
    Ok((StatusCode::CREATED, Json(cfg)))
}

/// `GET /api/v1/webhooks/{id}` — get a single webhook.
pub async fn get_webhook(
    State(manager): State<Arc<WebhookManager>>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, crate::error::ServerError> {
    let cfg = manager.get(&id).await?;
    Ok(Json(cfg))
}

/// `PUT /api/v1/webhooks/{id}` — update a webhook.
pub async fn update_webhook(
    State(manager): State<Arc<WebhookManager>>,
    Path(id): Path<String>,
    Json(body): Json<UpdateWebhookRequest>,
) -> Result<impl IntoResponse, crate::error::ServerError> {
    let cfg = manager.update(&id, body).await?;
    Ok(Json(cfg))
}

/// `DELETE /api/v1/webhooks/{id}` — remove a webhook.
pub async fn delete_webhook(
    State(manager): State<Arc<WebhookManager>>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, crate::error::ServerError> {
    manager.delete(&id).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// `GET /api/v1/webhooks/{id}/deliveries` — delivery history for a webhook.
pub async fn get_webhook_deliveries(
    State(manager): State<Arc<WebhookManager>>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, crate::error::ServerError> {
    // Verify the webhook exists first.
    manager.get(&id).await?;
    let deliveries = manager.deliveries_for(&id).await;
    Ok(Json(deliveries))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager() -> WebhookManager {
        WebhookManager::new_for_test()
    }

    fn reg(url: &str, events: &[&str]) -> RegisterWebhookRequest {
        RegisterWebhookRequest {
            url: url.to_string(),
            events: events.iter().map(|s| s.to_string()).collect(),
            secret: Some("test-secret".to_string()),
        }
    }

    #[tokio::test]
    async fn test_register_returns_config() {
        let m = make_manager();
        let cfg = m
            .register(reg("https://example.com/hook", &["media.uploaded"]))
            .await
            .expect("register");
        assert!(!cfg.id.is_empty());
        assert_eq!(cfg.url, "https://example.com/hook");
        assert!(cfg.active);
    }

    #[tokio::test]
    async fn test_register_rejects_empty_url() {
        let m = make_manager();
        assert!(m.register(reg("", &["media.uploaded"])).await.is_err());
    }

    #[tokio::test]
    async fn test_register_rejects_empty_events() {
        let m = make_manager();
        let req = RegisterWebhookRequest {
            url: "https://example.com".to_string(),
            events: vec![],
            secret: None,
        };
        assert!(m.register(req).await.is_err());
    }

    #[tokio::test]
    async fn test_list_returns_registered() {
        let m = make_manager();
        m.register(reg("https://a.com", &["transcode.completed"]))
            .await
            .expect("a");
        m.register(reg("https://b.com", &["media.deleted"]))
            .await
            .expect("b");
        assert_eq!(m.list().await.len(), 2);
    }

    #[tokio::test]
    async fn test_get_existing_webhook() {
        let m = make_manager();
        let cfg = m
            .register(reg("https://example.com", &["media.uploaded"]))
            .await
            .expect("reg");
        let fetched = m.get(&cfg.id).await.expect("get");
        assert_eq!(fetched.id, cfg.id);
    }

    #[tokio::test]
    async fn test_get_missing_webhook_errors() {
        let m = make_manager();
        assert!(m.get("nonexistent-id").await.is_err());
    }

    #[tokio::test]
    async fn test_update_webhook_url() {
        let m = make_manager();
        let cfg = m
            .register(reg("https://old.com", &["media.uploaded"]))
            .await
            .expect("reg");
        let updated = m
            .update(
                &cfg.id,
                UpdateWebhookRequest {
                    url: Some("https://new.com".to_string()),
                    events: None,
                    secret: None,
                    active: None,
                },
            )
            .await
            .expect("update");
        assert_eq!(updated.url, "https://new.com");
    }

    #[tokio::test]
    async fn test_delete_webhook() {
        let m = make_manager();
        let cfg = m
            .register(reg("https://example.com", &["media.deleted"]))
            .await
            .expect("reg");
        m.delete(&cfg.id).await.expect("delete");
        assert_eq!(m.list().await.len(), 0);
    }

    #[tokio::test]
    async fn test_delete_missing_errors() {
        let m = make_manager();
        assert!(m.delete("ghost-id").await.is_err());
    }

    #[test]
    fn test_sign_payload_deterministic() {
        let s1 = sign_payload("secret", "hello");
        let s2 = sign_payload("secret", "hello");
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_sign_payload_different_keys() {
        let s1 = sign_payload("key1", "data");
        let s2 = sign_payload("key2", "data");
        assert_ne!(s1, s2);
    }

    #[test]
    fn test_event_type_names() {
        assert_eq!(
            WebhookEvent::TranscodeCompleted {
                job_id: "j".into(),
                media_id: "m".into(),
                output_url: "u".into()
            }
            .event_type_name(),
            "transcode.completed"
        );
        assert_eq!(
            WebhookEvent::MediaDeleted {
                media_id: "m".into()
            }
            .event_type_name(),
            "media.deleted"
        );
    }

    #[test]
    fn test_delivery_status_serializes() {
        let j = serde_json::to_value(DeliveryStatus::Success).expect("serialize");
        assert_eq!(j, "success");
    }

    #[test]
    fn test_generate_secret_not_empty() {
        let s = generate_secret();
        assert!(!s.is_empty());
        assert_eq!(s.len(), 64); // 32 bytes → 64 hex chars
    }

    #[test]
    fn test_webhook_event_serializes_with_type_tag() {
        let ev = WebhookEvent::MediaUploaded {
            media_id: "m1".into(),
            filename: "foo.mp4".into(),
            size_bytes: 1024,
        };
        let j = serde_json::to_value(&ev).expect("serialize");
        assert_eq!(j["type"], "media_uploaded");
        assert_eq!(j["filename"], "foo.mp4");
    }
}
