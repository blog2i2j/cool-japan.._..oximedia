//! Notification system for job events

use crate::error::{BatchError, Result};
use crate::types::JobId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Notification event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationEvent {
    /// Job ID
    pub job_id: JobId,
    /// Job name
    pub job_name: String,
    /// Event type
    pub event_type: EventType,
    /// Timestamp
    pub timestamp: String,
    /// Additional data
    pub data: HashMap<String, String>,
}

/// Event types for notifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventType {
    /// Job submitted
    JobSubmitted,
    /// Job started
    JobStarted,
    /// Job completed
    JobCompleted,
    /// Job failed
    JobFailed,
    /// Job cancelled
    JobCancelled,
    /// Job progress update
    JobProgress {
        /// Progress percentage (0-100)
        progress: f64,
    },
}

/// Notification channel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationChannel {
    /// Email notification
    Email {
        /// Recipient email addresses
        to: Vec<String>,
        /// SMTP server
        smtp_server: String,
        /// SMTP port
        smtp_port: u16,
        /// From address
        from: String,
    },
    /// Webhook notification
    Webhook {
        /// Webhook URL
        url: String,
        /// HTTP headers
        headers: HashMap<String, String>,
    },
    /// Slack notification
    Slack {
        /// Webhook URL
        webhook_url: String,
    },
    /// Discord notification
    Discord {
        /// Webhook URL
        webhook_url: String,
    },
    /// Microsoft Teams notification
    Teams {
        /// Webhook URL
        webhook_url: String,
    },
}

/// Notification service
pub struct NotificationService {
    channels: Vec<NotificationChannel>,
    enabled_events: Vec<EventType>,
}

impl NotificationService {
    /// Create a new notification service
    #[must_use]
    pub fn new() -> Self {
        Self {
            channels: Vec::new(),
            enabled_events: Vec::new(),
        }
    }

    /// Add a notification channel
    pub fn add_channel(&mut self, channel: NotificationChannel) {
        self.channels.push(channel);
    }

    /// Enable event type
    pub fn enable_event(&mut self, event_type: EventType) {
        self.enabled_events.push(event_type);
    }

    /// Send notification
    ///
    /// # Arguments
    ///
    /// * `event` - Notification event
    ///
    /// # Errors
    ///
    /// Returns an error if notification fails
    pub async fn send(&self, event: NotificationEvent) -> Result<()> {
        for channel in &self.channels {
            if let Err(e) = self.send_to_channel(channel, &event).await {
                tracing::error!("Failed to send notification: {}", e);
            }
        }

        Ok(())
    }

    async fn send_to_channel(
        &self,
        channel: &NotificationChannel,
        event: &NotificationEvent,
    ) -> Result<()> {
        match channel {
            NotificationChannel::Email {
                to,
                smtp_server,
                smtp_port,
                from,
            } => {
                self.send_email(to, from, smtp_server, *smtp_port, event)
                    .await
            }
            NotificationChannel::Webhook { url, headers: _ } => {
                tracing::info!("Sending webhook notification to {}", url);
                self.send_webhook(url, event).await
            }
            NotificationChannel::Slack { webhook_url } => {
                tracing::info!("Sending Slack notification");
                self.send_slack(webhook_url, event).await
            }
            NotificationChannel::Discord { webhook_url } => {
                tracing::info!("Sending Discord notification");
                self.send_discord(webhook_url, event).await
            }
            NotificationChannel::Teams { webhook_url } => {
                tracing::info!("Sending Teams notification");
                self.send_teams(webhook_url, event).await
            }
        }
    }

    /// Send an email notification.
    ///
    /// This implementation constructs an RFC 5322 compliant email body and
    /// logs the full message at `info!` level rather than opening a live SMTP
    /// connection.  This is intentional: pure-Rust SMTP implementations
    /// require blocking I/O and TLS dependencies that are not yet part of the
    /// workspace.  When a production SMTP integration is required, replace the
    /// body of this function with calls to an appropriate pure-Rust SMTP
    /// client (e.g. `lettre`) once it has been added to the workspace.
    async fn send_email(
        &self,
        to: &[String],
        from: &str,
        smtp_server: &str,
        smtp_port: u16,
        event: &NotificationEvent,
    ) -> Result<()> {
        let subject = format!(
            "[OxiMedia Batch] Job '{}' — {}",
            event.job_name,
            Self::event_type_label(&event.event_type),
        );

        let body = Self::format_email_body(event);

        tracing::info!(
            smtp_server = %smtp_server,
            smtp_port = %smtp_port,
            from = %from,
            to = ?to,
            subject = %subject,
            "Email notification queued"
        );
        tracing::info!(
            job_id = %event.job_id,
            job_name = %event.job_name,
            timestamp = %event.timestamp,
            email_body = %body,
            "Email notification body"
        );

        Ok(())
    }

    /// Format a human-readable label for an event type.
    fn event_type_label(event_type: &EventType) -> &'static str {
        match event_type {
            EventType::JobSubmitted => "Submitted",
            EventType::JobStarted => "Started",
            EventType::JobCompleted => "Completed",
            EventType::JobFailed => "Failed",
            EventType::JobCancelled => "Cancelled",
            EventType::JobProgress { .. } => "Progress Update",
        }
    }

    /// Build a plain-text RFC 5322 email body for the given event.
    fn format_email_body(event: &NotificationEvent) -> String {
        let status_line = match &event.event_type {
            EventType::JobSubmitted => "The job has been submitted for processing.".to_string(),
            EventType::JobStarted => "The job has started processing.".to_string(),
            EventType::JobCompleted => "The job completed successfully.".to_string(),
            EventType::JobFailed => "The job encountered an error and failed.".to_string(),
            EventType::JobCancelled => "The job was cancelled.".to_string(),
            EventType::JobProgress { progress } => {
                format!("Job progress: {progress:.1}%")
            }
        };

        let mut lines = vec![
            format!("Job: {}", event.job_name),
            format!("Job ID: {}", event.job_id),
            format!("Timestamp: {}", event.timestamp),
            String::new(),
            status_line,
        ];

        if !event.data.is_empty() {
            lines.push(String::new());
            lines.push("Additional information:".to_string());
            let mut pairs: Vec<(&String, &String)> = event.data.iter().collect();
            pairs.sort_by_key(|(k, _)| k.as_str());
            for (key, value) in pairs {
                lines.push(format!("  {key}: {value}"));
            }
        }

        lines.join("\n")
    }

    async fn send_webhook(&self, url: &str, event: &NotificationEvent) -> Result<()> {
        let client = reqwest::Client::new();
        let payload = serde_json::to_string(event)?;

        let response = client
            .post(url)
            .header("Content-Type", "application/json")
            .body(payload)
            .send()
            .await
            .map_err(|e| BatchError::ApiError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(BatchError::ApiError(format!(
                "Webhook returned status: {}",
                response.status()
            )));
        }

        Ok(())
    }

    async fn send_slack(&self, webhook_url: &str, event: &NotificationEvent) -> Result<()> {
        let client = reqwest::Client::new();

        let message = Self::format_slack_message(event);
        let payload = serde_json::json!({
            "text": message
        });

        let response = client
            .post(webhook_url)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|e| BatchError::ApiError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(BatchError::ApiError(format!(
                "Slack webhook returned status: {}",
                response.status()
            )));
        }

        Ok(())
    }

    async fn send_discord(&self, webhook_url: &str, event: &NotificationEvent) -> Result<()> {
        let client = reqwest::Client::new();

        let message = Self::format_discord_message(event);
        let payload = serde_json::json!({
            "content": message
        });

        let response = client
            .post(webhook_url)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|e| BatchError::ApiError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(BatchError::ApiError(format!(
                "Discord webhook returned status: {}",
                response.status()
            )));
        }

        Ok(())
    }

    async fn send_teams(&self, webhook_url: &str, event: &NotificationEvent) -> Result<()> {
        let client = reqwest::Client::new();

        let message = Self::format_teams_message(event);
        let payload = serde_json::json!({
            "@type": "MessageCard",
            "@context": "https://schema.org/extensions",
            "summary": "Job Notification",
            "sections": [{
                "activityTitle": message,
                "facts": []
            }]
        });

        let response = client
            .post(webhook_url)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|e| BatchError::ApiError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(BatchError::ApiError(format!(
                "Teams webhook returned status: {}",
                response.status()
            )));
        }

        Ok(())
    }

    fn format_slack_message(event: &NotificationEvent) -> String {
        match &event.event_type {
            EventType::JobSubmitted => {
                format!("Job '{}' submitted", event.job_name)
            }
            EventType::JobStarted => {
                format!("Job '{}' started", event.job_name)
            }
            EventType::JobCompleted => {
                format!("Job '{}' completed successfully", event.job_name)
            }
            EventType::JobFailed => {
                format!("Job '{}' failed", event.job_name)
            }
            EventType::JobCancelled => {
                format!("Job '{}' cancelled", event.job_name)
            }
            EventType::JobProgress { progress } => {
                format!("Job '{}' progress: {:.1}%", event.job_name, progress)
            }
        }
    }

    fn format_discord_message(event: &NotificationEvent) -> String {
        Self::format_slack_message(event)
    }

    fn format_teams_message(event: &NotificationEvent) -> String {
        Self::format_slack_message(event)
    }
}

impl Default for NotificationService {
    fn default() -> Self {
        Self::new()
    }
}

/// Notification builder for fluent configuration
pub struct NotificationBuilder {
    service: NotificationService,
}

impl NotificationBuilder {
    /// Create a new notification builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            service: NotificationService::new(),
        }
    }

    /// Add email channel
    #[must_use]
    pub fn with_email(
        mut self,
        to: Vec<String>,
        smtp_server: String,
        smtp_port: u16,
        from: String,
    ) -> Self {
        self.service.add_channel(NotificationChannel::Email {
            to,
            smtp_server,
            smtp_port,
            from,
        });
        self
    }

    /// Add webhook channel
    #[must_use]
    pub fn with_webhook(mut self, url: String, headers: HashMap<String, String>) -> Self {
        self.service
            .add_channel(NotificationChannel::Webhook { url, headers });
        self
    }

    /// Add Slack channel
    #[must_use]
    pub fn with_slack(mut self, webhook_url: String) -> Self {
        self.service
            .add_channel(NotificationChannel::Slack { webhook_url });
        self
    }

    /// Add Discord channel
    #[must_use]
    pub fn with_discord(mut self, webhook_url: String) -> Self {
        self.service
            .add_channel(NotificationChannel::Discord { webhook_url });
        self
    }

    /// Add Teams channel
    #[must_use]
    pub fn with_teams(mut self, webhook_url: String) -> Self {
        self.service
            .add_channel(NotificationChannel::Teams { webhook_url });
        self
    }

    /// Build the notification service
    #[must_use]
    pub fn build(self) -> NotificationService {
        self.service
    }
}

impl Default for NotificationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_notification_service_creation() {
        let service = NotificationService::new();
        assert_eq!(service.channels.len(), 0);
    }

    #[test]
    fn test_add_email_channel() {
        let mut service = NotificationService::new();
        service.add_channel(NotificationChannel::Email {
            to: vec!["test@example.com".to_string()],
            smtp_server: "smtp.example.com".to_string(),
            smtp_port: 587,
            from: "noreply@example.com".to_string(),
        });

        assert_eq!(service.channels.len(), 1);
    }

    #[test]
    fn test_add_webhook_channel() {
        let mut service = NotificationService::new();
        service.add_channel(NotificationChannel::Webhook {
            url: "https://example.com/webhook".to_string(),
            headers: HashMap::new(),
        });

        assert_eq!(service.channels.len(), 1);
    }

    #[test]
    fn test_notification_builder() {
        let service = NotificationBuilder::new()
            .with_slack("https://slack.com/webhook".to_string())
            .with_discord("https://discord.com/webhook".to_string())
            .build();

        assert_eq!(service.channels.len(), 2);
    }

    #[test]
    fn test_format_slack_message() {
        let _service = NotificationService::new();
        let event = NotificationEvent {
            job_id: JobId::new(),
            job_name: "test-job".to_string(),
            event_type: EventType::JobCompleted,
            timestamp: chrono::Utc::now().to_rfc3339(),
            data: HashMap::new(),
        };

        let message = NotificationService::format_slack_message(&event);
        assert!(message.contains("test-job"));
        assert!(message.contains("completed"));
    }
}
