//! Webhook and event notification system
//!
//! Provides event-driven notifications for:
//! - Asset lifecycle events
//! - Workflow status changes
//! - Ingest completion
//! - Proxy generation completion
//! - Custom events
//! - HTTP webhooks
//! - Email notifications

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use uuid::Uuid;

use crate::database::Database;
use crate::{MamError, Result};

/// Webhook manager handles event notifications
pub struct WebhookManager {
    db: Arc<Database>,
    /// Event channel for internal event distribution
    event_tx: mpsc::UnboundedSender<Event>,
    /// Active webhook subscriptions
    webhooks: Arc<RwLock<HashMap<Uuid, Webhook>>>,
}

/// Webhook subscription
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Webhook {
    pub id: Uuid,
    pub name: String,
    pub url: String,
    pub event_types: Vec<String>,
    pub secret: Option<String>,
    pub is_active: bool,
    pub retry_count: i32,
    pub timeout_seconds: i32,
    pub created_by: Option<Uuid>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Event type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventType {
    // Asset events
    /// Asset created
    AssetCreated,
    /// Asset updated
    AssetUpdated,
    /// Asset deleted
    AssetDeleted,
    /// Asset status changed
    AssetStatusChanged,

    // Ingest events
    /// Ingest started
    IngestStarted,
    /// Ingest completed
    IngestCompleted,
    /// Ingest failed
    IngestFailed,

    // Proxy events
    /// Proxy generation started
    ProxyStarted,
    /// Proxy generation completed
    ProxyCompleted,
    /// Proxy generation failed
    ProxyFailed,

    // Workflow events
    /// Workflow created
    WorkflowCreated,
    /// Workflow updated
    WorkflowUpdated,
    /// Workflow completed
    WorkflowCompleted,
    /// Workflow approved
    WorkflowApproved,
    /// Workflow rejected
    WorkflowRejected,

    // Collection events
    /// Collection created
    CollectionCreated,
    /// Collection updated
    CollectionUpdated,
    /// Collection deleted
    CollectionDeleted,

    // User events
    /// User logged in
    UserLoggedIn,
    /// User created
    UserCreated,
    /// User updated
    UserUpdated,

    // Storage events
    /// File uploaded
    FileUploaded,
    /// File deleted
    FileDeleted,
    /// Storage tier changed
    StorageTierChanged,

    // Custom event
    /// Custom event
    Custom,
}

impl EventType {
    /// Convert to string
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::AssetCreated => "asset.created",
            Self::AssetUpdated => "asset.updated",
            Self::AssetDeleted => "asset.deleted",
            Self::AssetStatusChanged => "asset.status_changed",
            Self::IngestStarted => "ingest.started",
            Self::IngestCompleted => "ingest.completed",
            Self::IngestFailed => "ingest.failed",
            Self::ProxyStarted => "proxy.started",
            Self::ProxyCompleted => "proxy.completed",
            Self::ProxyFailed => "proxy.failed",
            Self::WorkflowCreated => "workflow.created",
            Self::WorkflowUpdated => "workflow.updated",
            Self::WorkflowCompleted => "workflow.completed",
            Self::WorkflowApproved => "workflow.approved",
            Self::WorkflowRejected => "workflow.rejected",
            Self::CollectionCreated => "collection.created",
            Self::CollectionUpdated => "collection.updated",
            Self::CollectionDeleted => "collection.deleted",
            Self::UserLoggedIn => "user.logged_in",
            Self::UserCreated => "user.created",
            Self::UserUpdated => "user.updated",
            Self::FileUploaded => "file.uploaded",
            Self::FileDeleted => "file.deleted",
            Self::StorageTierChanged => "storage.tier_changed",
            Self::Custom => "custom",
        }
    }
}

impl std::str::FromStr for EventType {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "asset.created" => Ok(Self::AssetCreated),
            "asset.updated" => Ok(Self::AssetUpdated),
            "asset.deleted" => Ok(Self::AssetDeleted),
            "asset.status_changed" => Ok(Self::AssetStatusChanged),
            "ingest.started" => Ok(Self::IngestStarted),
            "ingest.completed" => Ok(Self::IngestCompleted),
            "ingest.failed" => Ok(Self::IngestFailed),
            "proxy.started" => Ok(Self::ProxyStarted),
            "proxy.completed" => Ok(Self::ProxyCompleted),
            "proxy.failed" => Ok(Self::ProxyFailed),
            "workflow.created" => Ok(Self::WorkflowCreated),
            "workflow.updated" => Ok(Self::WorkflowUpdated),
            "workflow.completed" => Ok(Self::WorkflowCompleted),
            "workflow.approved" => Ok(Self::WorkflowApproved),
            "workflow.rejected" => Ok(Self::WorkflowRejected),
            "collection.created" => Ok(Self::CollectionCreated),
            "collection.updated" => Ok(Self::CollectionUpdated),
            "collection.deleted" => Ok(Self::CollectionDeleted),
            "user.logged_in" => Ok(Self::UserLoggedIn),
            "user.created" => Ok(Self::UserCreated),
            "user.updated" => Ok(Self::UserUpdated),
            "file.uploaded" => Ok(Self::FileUploaded),
            "file.deleted" => Ok(Self::FileDeleted),
            "storage.tier_changed" => Ok(Self::StorageTierChanged),
            "custom" => Ok(Self::Custom),
            _ => Err(format!("Invalid event type: {s}")),
        }
    }
}

/// Event data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: Uuid,
    pub event_type: EventType,
    pub resource_id: Option<Uuid>,
    pub resource_type: Option<String>,
    pub user_id: Option<Uuid>,
    pub data: serde_json::Value,
    pub timestamp: DateTime<Utc>,
}

/// Webhook delivery attempt
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct WebhookDelivery {
    pub id: Uuid,
    pub webhook_id: Uuid,
    pub event_id: Uuid,
    pub status: String,
    pub response_status: Option<i32>,
    pub response_body: Option<String>,
    pub error_message: Option<String>,
    pub attempt: i32,
    pub delivered_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

/// Webhook delivery status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeliveryStatus {
    /// Pending delivery
    Pending,
    /// Successfully delivered
    Delivered,
    /// Failed to deliver
    Failed,
    /// Retrying delivery
    Retrying,
}

impl DeliveryStatus {
    /// Convert to string
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Delivered => "delivered",
            Self::Failed => "failed",
            Self::Retrying => "retrying",
        }
    }
}

/// Webhook creation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateWebhookRequest {
    pub name: String,
    pub url: String,
    pub event_types: Vec<String>,
    pub secret: Option<String>,
    pub timeout_seconds: Option<i32>,
}

impl WebhookManager {
    /// Create a new webhook manager
    #[must_use]
    pub fn new(db: Arc<Database>) -> Self {
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();
        let webhooks = Arc::new(RwLock::new(HashMap::new()));

        // Spawn event processor
        let webhooks_clone = Arc::clone(&webhooks);
        let db_clone = Arc::clone(&db);
        tokio::spawn(async move {
            while let Some(event) = event_rx.recv().await {
                Self::process_event_internal(
                    event,
                    Arc::clone(&webhooks_clone),
                    Arc::clone(&db_clone),
                )
                .await;
            }
        });

        Self {
            db,
            event_tx,
            webhooks,
        }
    }

    /// Emit an event
    ///
    /// # Errors
    ///
    /// Returns an error if event emission fails
    pub fn emit(&self, event: Event) -> Result<()> {
        self.event_tx
            .send(event)
            .map_err(|e| MamError::Internal(format!("Failed to emit event: {e}")))?;
        Ok(())
    }

    /// Process event (internal)
    async fn process_event_internal(
        event: Event,
        webhooks: Arc<RwLock<HashMap<Uuid, Webhook>>>,
        db: Arc<Database>,
    ) {
        // Store event in database
        if let Err(e) = Self::store_event(&event, &db).await {
            tracing::error!("Failed to store event: {}", e);
            return;
        }

        // Find matching webhooks
        let webhooks_map = webhooks.read().await;
        let matching_webhooks: Vec<Webhook> = webhooks_map
            .values()
            .filter(|w| {
                w.is_active
                    && w.event_types
                        .contains(&event.event_type.as_str().to_string())
            })
            .cloned()
            .collect();

        drop(webhooks_map);

        // Deliver to each webhook
        for webhook in matching_webhooks {
            tokio::spawn(Self::deliver_webhook(event.clone(), webhook, db.clone()));
        }
    }

    /// Store event in database
    async fn store_event(event: &Event, db: &Database) -> Result<()> {
        sqlx::query(
            "INSERT INTO events
             (id, event_type, resource_id, resource_type, user_id, data, timestamp)
             VALUES ($1, $2, $3, $4, $5, $6, $7)",
        )
        .bind(event.id)
        .bind(event.event_type.as_str())
        .bind(event.resource_id)
        .bind(&event.resource_type)
        .bind(event.user_id)
        .bind(&event.data)
        .bind(event.timestamp)
        .execute(db.pool())
        .await?;

        Ok(())
    }

    /// Deliver webhook
    async fn deliver_webhook(event: Event, webhook: Webhook, db: Arc<Database>) {
        let delivery_id = Uuid::new_v4();
        let mut attempt = 1;
        let max_retries = webhook.retry_count;

        loop {
            // Create delivery record
            if let Err(e) = sqlx::query(
                "INSERT INTO webhook_deliveries
                 (id, webhook_id, event_id, status, attempt, created_at)
                 VALUES ($1, $2, $3, 'pending', $4, NOW())",
            )
            .bind(delivery_id)
            .bind(webhook.id)
            .bind(event.id)
            .bind(attempt)
            .execute(db.pool())
            .await
            {
                tracing::error!("Failed to create delivery record: {}", e);
                return;
            }

            // Build webhook payload
            let payload = serde_json::json!({
                "event_id": event.id,
                "event_type": event.event_type.as_str(),
                "resource_id": event.resource_id,
                "resource_type": event.resource_type,
                "user_id": event.user_id,
                "data": event.data,
                "timestamp": event.timestamp,
            });

            // Send HTTP request
            let client = reqwest::Client::new();
            let mut request =
                client
                    .post(&webhook.url)
                    .json(&payload)
                    .timeout(std::time::Duration::from_secs(
                        webhook.timeout_seconds as u64,
                    ));

            // Add signature if secret is set
            if let Some(secret) = &webhook.secret {
                let signature = Self::calculate_signature(&payload, secret);
                request = request.header("X-Webhook-Signature", signature);
            }

            match request.send().await {
                Ok(response) => {
                    let status = response.status().as_u16() as i32;
                    let body = response.text().await.ok();

                    if (200..300).contains(&(status as u16)) {
                        // Success
                        let _ = sqlx::query(
                            "UPDATE webhook_deliveries
                             SET status = 'delivered', response_status = $2, response_body = $3, delivered_at = NOW()
                             WHERE id = $1",
                        )
                        .bind(delivery_id)
                        .bind(status)
                        .bind(body)
                        .execute(db.pool())
                        .await;

                        return;
                    } else {
                        // HTTP error
                        let _ = sqlx::query(
                            "UPDATE webhook_deliveries
                             SET status = 'failed', response_status = $2, response_body = $3, error_message = $4
                             WHERE id = $1",
                        )
                        .bind(delivery_id)
                        .bind(status)
                        .bind(&body)
                        .bind(format!("HTTP error: {status}"))
                        .execute(db.pool())
                        .await;
                    }
                }
                Err(e) => {
                    // Network error
                    let _ = sqlx::query(
                        "UPDATE webhook_deliveries
                         SET status = 'failed', error_message = $2
                         WHERE id = $1",
                    )
                    .bind(delivery_id)
                    .bind(e.to_string())
                    .execute(db.pool())
                    .await;
                }
            }

            // Retry logic
            attempt += 1;
            if attempt > max_retries {
                tracing::error!(
                    "Webhook delivery failed after {} attempts: {}",
                    max_retries,
                    webhook.url
                );
                return;
            }

            // Exponential backoff
            let delay = std::time::Duration::from_secs(2_u64.pow((attempt - 2) as u32));
            tokio::time::sleep(delay).await;
        }
    }

    /// Calculate webhook signature
    fn calculate_signature(payload: &serde_json::Value, secret: &str) -> String {
        use sha2::{Digest, Sha256};

        let payload_str = serde_json::to_string(payload).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(secret.as_bytes());
        hasher.update(payload_str.as_bytes());
        let result = hasher.finalize();
        result.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Create a webhook subscription
    ///
    /// # Errors
    ///
    /// Returns an error if creation fails
    pub async fn create_webhook(
        &self,
        req: CreateWebhookRequest,
        created_by: Option<Uuid>,
    ) -> Result<Webhook> {
        let webhook = sqlx::query_as::<_, Webhook>(
            "INSERT INTO webhooks
             (id, name, url, event_types, secret, is_active, retry_count, timeout_seconds, created_by, created_at, updated_at)
             VALUES ($1, $2, $3, $4, $5, true, 3, $6, $7, NOW(), NOW())
             RETURNING *",
        )
        .bind(Uuid::new_v4())
        .bind(&req.name)
        .bind(&req.url)
        .bind(&req.event_types)
        .bind(&req.secret)
        .bind(req.timeout_seconds.unwrap_or(30))
        .bind(created_by)
        .fetch_one(self.db.pool())
        .await?;

        // Add to active webhooks
        self.webhooks
            .write()
            .await
            .insert(webhook.id, webhook.clone());

        Ok(webhook)
    }

    /// Get webhook by ID
    ///
    /// # Errors
    ///
    /// Returns an error if webhook not found
    pub async fn get_webhook(&self, webhook_id: Uuid) -> Result<Webhook> {
        let webhook = sqlx::query_as::<_, Webhook>("SELECT * FROM webhooks WHERE id = $1")
            .bind(webhook_id)
            .fetch_one(self.db.pool())
            .await?;

        Ok(webhook)
    }

    /// List all webhooks
    ///
    /// # Errors
    ///
    /// Returns an error if query fails
    pub async fn list_webhooks(&self) -> Result<Vec<Webhook>> {
        let webhooks =
            sqlx::query_as::<_, Webhook>("SELECT * FROM webhooks ORDER BY created_at DESC")
                .fetch_all(self.db.pool())
                .await?;

        Ok(webhooks)
    }

    /// Delete webhook
    ///
    /// # Errors
    ///
    /// Returns an error if deletion fails
    pub async fn delete_webhook(&self, webhook_id: Uuid) -> Result<()> {
        sqlx::query("DELETE FROM webhooks WHERE id = $1")
            .bind(webhook_id)
            .execute(self.db.pool())
            .await?;

        self.webhooks.write().await.remove(&webhook_id);

        Ok(())
    }

    /// Get webhook deliveries
    ///
    /// # Errors
    ///
    /// Returns an error if query fails
    pub async fn get_webhook_deliveries(
        &self,
        webhook_id: Uuid,
        limit: i64,
    ) -> Result<Vec<WebhookDelivery>> {
        let deliveries = sqlx::query_as::<_, WebhookDelivery>(
            "SELECT * FROM webhook_deliveries WHERE webhook_id = $1 ORDER BY created_at DESC LIMIT $2",
        )
        .bind(webhook_id)
        .bind(limit)
        .fetch_all(self.db.pool())
        .await?;

        Ok(deliveries)
    }

    /// Load all webhooks from database
    ///
    /// # Errors
    ///
    /// Returns an error if query fails
    pub async fn load_webhooks(&self) -> Result<()> {
        let webhooks = self.list_webhooks().await?;

        let mut webhooks_map = self.webhooks.write().await;
        webhooks_map.clear();

        for webhook in webhooks {
            webhooks_map.insert(webhook.id, webhook);
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Retry with exponential backoff engine
// ---------------------------------------------------------------------------

/// Configuration for retry with exponential backoff.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts (not counting the initial attempt).
    pub max_retries: u32,
    /// Initial backoff delay in milliseconds.
    pub initial_backoff_ms: u64,
    /// Multiplier applied to the backoff after each retry (typically 2.0).
    pub backoff_multiplier: f64,
    /// Maximum backoff delay in milliseconds (cap).
    pub max_backoff_ms: u64,
    /// Optional jitter factor (0.0 = none, 1.0 = full jitter up to backoff).
    /// Applied deterministically using attempt number as seed to stay pure.
    pub jitter_factor: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 5,
            initial_backoff_ms: 1000,
            backoff_multiplier: 2.0,
            max_backoff_ms: 60_000,
            jitter_factor: 0.0,
        }
    }
}

impl RetryPolicy {
    /// Create a new retry policy with the given max retries.
    #[must_use]
    pub fn new(max_retries: u32) -> Self {
        Self {
            max_retries,
            ..Default::default()
        }
    }

    /// Set initial backoff in milliseconds.
    #[must_use]
    pub fn with_initial_backoff_ms(mut self, ms: u64) -> Self {
        self.initial_backoff_ms = ms;
        self
    }

    /// Set the backoff multiplier.
    #[must_use]
    pub fn with_multiplier(mut self, m: f64) -> Self {
        self.backoff_multiplier = m;
        self
    }

    /// Set the maximum backoff cap in milliseconds.
    #[must_use]
    pub fn with_max_backoff_ms(mut self, ms: u64) -> Self {
        self.max_backoff_ms = ms;
        self
    }

    /// Set jitter factor (0.0..=1.0). Uses deterministic pseudo-jitter.
    #[must_use]
    pub fn with_jitter(mut self, factor: f64) -> Self {
        self.jitter_factor = factor.clamp(0.0, 1.0);
        self
    }

    /// Compute the backoff duration for the given attempt (0-indexed).
    #[must_use]
    pub fn backoff_for_attempt(&self, attempt: u32) -> std::time::Duration {
        let base = self.initial_backoff_ms as f64 * self.backoff_multiplier.powi(attempt as i32);
        let capped = base.min(self.max_backoff_ms as f64);

        // Deterministic pseudo-jitter based on attempt number.
        let jitter = if self.jitter_factor > 0.0 {
            let seed = (attempt as f64 * 0.618_033_988_749_895).fract();
            capped * self.jitter_factor * seed
        } else {
            0.0
        };

        let total_ms = (capped + jitter).min(self.max_backoff_ms as f64 * 2.0);
        std::time::Duration::from_millis(total_ms as u64)
    }
}

/// Outcome of a single delivery attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeliveryOutcome {
    /// Delivery succeeded.
    Success,
    /// Delivery failed with a retryable error.
    RetryableFailure(String),
    /// Delivery failed with a permanent (non-retryable) error.
    PermanentFailure(String),
}

/// Record of a single retry attempt.
#[derive(Debug, Clone)]
pub struct RetryAttempt {
    /// 1-indexed attempt number.
    pub attempt: u32,
    /// Outcome of this attempt.
    pub outcome: DeliveryOutcome,
    /// Backoff that was (or would be) waited before this attempt.
    pub backoff: std::time::Duration,
}

/// The full result of a retry execution.
#[derive(Debug, Clone)]
pub struct RetryResult {
    /// Whether the overall delivery eventually succeeded.
    pub success: bool,
    /// Total number of attempts made.
    pub total_attempts: u32,
    /// Detail of each attempt.
    pub attempts: Vec<RetryAttempt>,
    /// Final outcome.
    pub final_outcome: DeliveryOutcome,
}

/// A synchronous retry executor that computes the retry schedule and collects
/// attempt results.  Actual waiting is handled by the caller (e.g.
/// `tokio::time::sleep`).  This struct computes the plan and records outcomes.
pub struct RetryExecutor {
    policy: RetryPolicy,
}

impl RetryExecutor {
    /// Create a new executor with the given policy.
    #[must_use]
    pub fn new(policy: RetryPolicy) -> Self {
        Self { policy }
    }

    /// Return a reference to the retry policy.
    #[must_use]
    pub fn policy(&self) -> &RetryPolicy {
        &self.policy
    }

    /// Compute the full backoff schedule (attempt 0 .. max_retries).
    #[must_use]
    pub fn schedule(&self) -> Vec<std::time::Duration> {
        (0..=self.policy.max_retries)
            .map(|i| self.policy.backoff_for_attempt(i))
            .collect()
    }

    /// Record a series of delivery outcomes and produce a [`RetryResult`].
    ///
    /// `outcomes` should contain up to `max_retries + 1` items.  The executor
    /// stops at the first `Success` or `PermanentFailure`, or when retries
    /// are exhausted.
    #[must_use]
    pub fn execute(&self, outcomes: &[DeliveryOutcome]) -> RetryResult {
        let mut attempts = Vec::new();
        let max = (self.policy.max_retries + 1) as usize;

        for (i, outcome) in outcomes.iter().take(max).enumerate() {
            let backoff = if i == 0 {
                std::time::Duration::ZERO
            } else {
                self.policy.backoff_for_attempt((i - 1) as u32)
            };

            attempts.push(RetryAttempt {
                attempt: (i + 1) as u32,
                outcome: outcome.clone(),
                backoff,
            });

            match outcome {
                DeliveryOutcome::Success => {
                    return RetryResult {
                        success: true,
                        total_attempts: (i + 1) as u32,
                        final_outcome: DeliveryOutcome::Success,
                        attempts,
                    };
                }
                DeliveryOutcome::PermanentFailure(_) => {
                    return RetryResult {
                        success: false,
                        total_attempts: (i + 1) as u32,
                        final_outcome: outcome.clone(),
                        attempts,
                    };
                }
                DeliveryOutcome::RetryableFailure(_) => {
                    // Continue to next attempt.
                }
            }
        }

        let final_outcome = attempts
            .last()
            .map(|a| a.outcome.clone())
            .unwrap_or_else(|| DeliveryOutcome::PermanentFailure("no attempts".to_string()));

        RetryResult {
            success: false,
            total_attempts: attempts.len() as u32,
            final_outcome,
            attempts,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_type_as_str() {
        assert_eq!(EventType::AssetCreated.as_str(), "asset.created");
        assert_eq!(EventType::AssetUpdated.as_str(), "asset.updated");
        assert_eq!(EventType::ProxyCompleted.as_str(), "proxy.completed");
    }

    #[test]
    fn test_event_type_from_str() {
        use std::str::FromStr;
        assert_eq!(
            EventType::from_str("asset.created").ok(),
            Some(EventType::AssetCreated)
        );
        assert_eq!(
            EventType::from_str("proxy.completed").ok(),
            Some(EventType::ProxyCompleted)
        );
        assert!(EventType::from_str("invalid").is_err());
    }

    #[test]
    fn test_delivery_status_as_str() {
        assert_eq!(DeliveryStatus::Pending.as_str(), "pending");
        assert_eq!(DeliveryStatus::Delivered.as_str(), "delivered");
        assert_eq!(DeliveryStatus::Failed.as_str(), "failed");
    }

    #[test]
    fn test_event_serialization() {
        let event = Event {
            id: Uuid::new_v4(),
            event_type: EventType::AssetCreated,
            resource_id: Some(Uuid::new_v4()),
            resource_type: Some("asset".to_string()),
            user_id: None,
            data: serde_json::json!({"test": "data"}),
            timestamp: Utc::now(),
        };

        let json = serde_json::to_string(&event).expect("should succeed in test");
        let deserialized: Event = serde_json::from_str(&json).expect("should succeed in test");

        assert_eq!(deserialized.event_type, EventType::AssetCreated);
    }

    #[test]
    fn test_create_webhook_request() {
        let req = CreateWebhookRequest {
            name: "Test Webhook".to_string(),
            url: "https://example.com/webhook".to_string(),
            event_types: vec!["asset.created".to_string(), "asset.updated".to_string()],
            secret: Some("secret123".to_string()),
            timeout_seconds: Some(30),
        };

        assert_eq!(req.name, "Test Webhook");
        assert_eq!(req.event_types.len(), 2);
    }

    // -----------------------------------------------------------------------
    // RetryPolicy tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_retry_policy_default() {
        let p = RetryPolicy::default();
        assert_eq!(p.max_retries, 5);
        assert_eq!(p.initial_backoff_ms, 1000);
        assert!((p.backoff_multiplier - 2.0).abs() < f64::EPSILON);
        assert_eq!(p.max_backoff_ms, 60_000);
        assert!((p.jitter_factor - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_retry_policy_builder() {
        let p = RetryPolicy::new(3)
            .with_initial_backoff_ms(500)
            .with_multiplier(3.0)
            .with_max_backoff_ms(30_000)
            .with_jitter(0.5);
        assert_eq!(p.max_retries, 3);
        assert_eq!(p.initial_backoff_ms, 500);
        assert!((p.backoff_multiplier - 3.0).abs() < f64::EPSILON);
        assert_eq!(p.max_backoff_ms, 30_000);
        assert!((p.jitter_factor - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_backoff_exponential_growth() {
        let p = RetryPolicy::new(5)
            .with_initial_backoff_ms(1000)
            .with_multiplier(2.0)
            .with_max_backoff_ms(100_000);

        // attempt 0: 1000ms, 1: 2000ms, 2: 4000ms, 3: 8000ms
        assert_eq!(p.backoff_for_attempt(0).as_millis(), 1000);
        assert_eq!(p.backoff_for_attempt(1).as_millis(), 2000);
        assert_eq!(p.backoff_for_attempt(2).as_millis(), 4000);
        assert_eq!(p.backoff_for_attempt(3).as_millis(), 8000);
    }

    #[test]
    fn test_backoff_capped() {
        let p = RetryPolicy::new(10)
            .with_initial_backoff_ms(1000)
            .with_multiplier(2.0)
            .with_max_backoff_ms(5000);

        // 2^4 * 1000 = 16000 > 5000 cap
        let d = p.backoff_for_attempt(4);
        assert_eq!(d.as_millis(), 5000);
    }

    #[test]
    fn test_backoff_with_jitter() {
        let p = RetryPolicy::new(3)
            .with_initial_backoff_ms(1000)
            .with_multiplier(2.0)
            .with_max_backoff_ms(100_000)
            .with_jitter(0.5);

        let base = p.backoff_for_attempt(0);
        // With jitter, should be >= 1000ms (base might have jitter added).
        assert!(base.as_millis() >= 1000);
    }

    #[test]
    fn test_jitter_clamped() {
        let p = RetryPolicy::new(1).with_jitter(5.0); // should clamp to 1.0
        assert!((p.jitter_factor - 1.0).abs() < f64::EPSILON);
        let p2 = RetryPolicy::new(1).with_jitter(-1.0); // should clamp to 0.0
        assert!((p2.jitter_factor - 0.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // RetryExecutor tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_schedule_length() {
        let policy = RetryPolicy::new(3);
        let executor = RetryExecutor::new(policy);
        let sched = executor.schedule();
        assert_eq!(sched.len(), 4); // initial + 3 retries
    }

    #[test]
    fn test_execute_immediate_success() {
        let executor = RetryExecutor::new(RetryPolicy::new(3));
        let outcomes = vec![DeliveryOutcome::Success];
        let result = executor.execute(&outcomes);
        assert!(result.success);
        assert_eq!(result.total_attempts, 1);
        assert_eq!(result.final_outcome, DeliveryOutcome::Success);
    }

    #[test]
    fn test_execute_success_after_retries() {
        let executor = RetryExecutor::new(RetryPolicy::new(5));
        let outcomes = vec![
            DeliveryOutcome::RetryableFailure("timeout".to_string()),
            DeliveryOutcome::RetryableFailure("502".to_string()),
            DeliveryOutcome::Success,
        ];
        let result = executor.execute(&outcomes);
        assert!(result.success);
        assert_eq!(result.total_attempts, 3);
        assert_eq!(result.attempts.len(), 3);
        // First attempt has zero backoff.
        assert_eq!(result.attempts[0].backoff, std::time::Duration::ZERO);
        // Second attempt has backoff.
        assert!(result.attempts[1].backoff > std::time::Duration::ZERO);
    }

    #[test]
    fn test_execute_exhausted_retries() {
        let executor = RetryExecutor::new(RetryPolicy::new(2));
        let outcomes = vec![
            DeliveryOutcome::RetryableFailure("fail1".to_string()),
            DeliveryOutcome::RetryableFailure("fail2".to_string()),
            DeliveryOutcome::RetryableFailure("fail3".to_string()),
        ];
        let result = executor.execute(&outcomes);
        assert!(!result.success);
        assert_eq!(result.total_attempts, 3); // initial + 2 retries
        assert_eq!(
            result.final_outcome,
            DeliveryOutcome::RetryableFailure("fail3".to_string())
        );
    }

    #[test]
    fn test_execute_permanent_failure_stops_retries() {
        let executor = RetryExecutor::new(RetryPolicy::new(5));
        let outcomes = vec![
            DeliveryOutcome::RetryableFailure("timeout".to_string()),
            DeliveryOutcome::PermanentFailure("404 not found".to_string()),
            DeliveryOutcome::Success, // should never be reached
        ];
        let result = executor.execute(&outcomes);
        assert!(!result.success);
        assert_eq!(result.total_attempts, 2);
        assert_eq!(
            result.final_outcome,
            DeliveryOutcome::PermanentFailure("404 not found".to_string())
        );
    }

    #[test]
    fn test_execute_empty_outcomes() {
        let executor = RetryExecutor::new(RetryPolicy::new(3));
        let result = executor.execute(&[]);
        assert!(!result.success);
        assert_eq!(result.total_attempts, 0);
    }

    #[test]
    fn test_execute_backoff_increases() {
        let policy = RetryPolicy::new(4)
            .with_initial_backoff_ms(100)
            .with_multiplier(2.0);
        let executor = RetryExecutor::new(policy);
        let outcomes = vec![
            DeliveryOutcome::RetryableFailure("1".to_string()),
            DeliveryOutcome::RetryableFailure("2".to_string()),
            DeliveryOutcome::RetryableFailure("3".to_string()),
            DeliveryOutcome::RetryableFailure("4".to_string()),
            DeliveryOutcome::RetryableFailure("5".to_string()),
        ];
        let result = executor.execute(&outcomes);
        // Backoff: 0, 100, 200, 400, 800
        assert_eq!(result.attempts[0].backoff, std::time::Duration::ZERO);
        assert_eq!(result.attempts[1].backoff.as_millis(), 100);
        assert_eq!(result.attempts[2].backoff.as_millis(), 200);
        assert_eq!(result.attempts[3].backoff.as_millis(), 400);
        assert_eq!(result.attempts[4].backoff.as_millis(), 800);
    }

    #[test]
    fn test_retry_policy_serialization() {
        let policy = RetryPolicy::new(3).with_initial_backoff_ms(500);
        let json = serde_json::to_string(&policy).expect("should succeed in test");
        let deserialized: RetryPolicy =
            serde_json::from_str(&json).expect("should succeed in test");
        assert_eq!(deserialized.max_retries, 3);
        assert_eq!(deserialized.initial_backoff_ms, 500);
    }

    #[test]
    fn test_execute_fewer_outcomes_than_retries() {
        // If we provide fewer outcomes than max_retries+1, it stops early.
        let executor = RetryExecutor::new(RetryPolicy::new(10));
        let outcomes = vec![
            DeliveryOutcome::RetryableFailure("a".to_string()),
            DeliveryOutcome::RetryableFailure("b".to_string()),
        ];
        let result = executor.execute(&outcomes);
        assert!(!result.success);
        assert_eq!(result.total_attempts, 2);
    }
}
