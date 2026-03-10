//! File logging alert channel.

use crate::alert::channels::AlertChannel;
use crate::alert::Alert;
use crate::error::{MonitorError, MonitorResult};
use async_trait::async_trait;
use std::path::PathBuf;
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;

/// File alert channel.
///
/// Appends a structured log line for each alert to the configured file.
/// Line format: `[TIMESTAMP] [SEVERITY] RULE_NAME (METRIC=VALUE): message {labels}`
pub struct FileChannel {
    path: PathBuf,
}

impl FileChannel {
    /// Create a new file channel.
    #[must_use]
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    /// Format the alert as a single log line.
    fn format_line(alert: &Alert) -> String {
        let timestamp = alert.timestamp.format("%Y-%m-%dT%H:%M:%S%.3fZ");
        let mut line = format!(
            "[{}] [{}] {} ({}={:.4}): {}",
            timestamp,
            alert.severity,
            alert.rule_name,
            alert.metric_name,
            alert.metric_value,
            alert.message,
        );

        if let Some(threshold) = alert.threshold {
            line.push_str(&format!(" [threshold={threshold:.4}]"));
        }

        if !alert.labels.is_empty() {
            let label_str: Vec<String> = alert
                .labels
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect();
            line.push_str(&format!(" {{{}}}", label_str.join(", ")));
        }

        line.push_str(&format!(" state={:?} id={}", alert.state, alert.id));
        line.push('\n');
        line
    }
}

#[async_trait]
impl AlertChannel for FileChannel {
    async fn send(&self, alert: &Alert) -> MonitorResult<()> {
        // Ensure parent directory exists.
        if let Some(parent) = self.path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(MonitorError::Io)?;
        }

        let line = Self::format_line(alert);

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .await
            .map_err(|e| {
                MonitorError::Io(std::io::Error::new(
                    e.kind(),
                    format!("Failed to open alert log {:?}: {}", self.path, e),
                ))
            })?;

        file.write_all(line.as_bytes())
            .await
            .map_err(MonitorError::Io)?;

        file.flush().await.map_err(MonitorError::Io)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alert::severity::AlertSeverity;
    use crate::alert::Alert;

    #[test]
    fn test_format_line_basic() {
        let alert = Alert::new(
            "cpu_high",
            AlertSeverity::Warning,
            "CPU usage is high",
            "cpu.usage",
            95.5,
        )
        .with_threshold(90.0)
        .with_label("host", "server-1");

        let line = FileChannel::format_line(&alert);
        assert!(line.contains("[WARNING]"));
        assert!(line.contains("cpu_high"));
        assert!(line.contains("cpu.usage=95.5000"));
        assert!(line.contains("CPU usage is high"));
        assert!(line.contains("threshold=90.0000"));
        assert!(line.contains("host=server-1"));
        assert!(line.ends_with('\n'));
    }

    #[test]
    fn test_format_line_no_threshold() {
        let alert = Alert::new(
            "test_rule",
            AlertSeverity::Info,
            "Just an info",
            "test.metric",
            1.0,
        );

        let line = FileChannel::format_line(&alert);
        assert!(!line.contains("threshold"));
        assert!(line.contains("[INFO]"));
    }

    #[tokio::test]
    async fn test_send_writes_to_file() {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        let path = dir.path().join("alerts.log");

        let channel = FileChannel::new(path.clone());
        let alert = Alert::new(
            "test",
            AlertSeverity::Critical,
            "Critical error",
            "error.count",
            42.0,
        );

        channel.send(&alert).await.expect("failed to send");

        let contents = tokio::fs::read_to_string(&path)
            .await
            .expect("formatting should succeed");
        assert!(contents.contains("[CRITICAL]"));
        assert!(contents.contains("test"));
        assert!(contents.contains("Critical error"));
    }

    #[tokio::test]
    async fn test_send_appends() {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        let path = dir.path().join("alerts.log");

        let channel = FileChannel::new(path.clone());

        for i in 0..3u32 {
            let alert = Alert::new(
                format!("rule_{}", i),
                AlertSeverity::Info,
                format!("Message {}", i),
                "metric",
                f64::from(i),
            );
            channel.send(&alert).await.expect("failed to send");
        }

        let contents = tokio::fs::read_to_string(&path)
            .await
            .expect("formatting should succeed");
        assert_eq!(contents.lines().count(), 3);
    }
}
