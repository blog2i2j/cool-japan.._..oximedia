// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Real-time monitoring and metrics.

use crate::error::Result;
use crate::job::{JobId, JobState};
use crate::worker::{WorkerId, WorkerState};
use chrono::{DateTime, Utc};
use prometheus::{Counter, Gauge, Histogram, Registry};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Monitor configuration
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Metrics collection interval (seconds)
    pub collection_interval: u64,
    /// Retention period (hours)
    pub retention_hours: u32,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            collection_interval: 10,
            retention_hours: 24,
        }
    }
}

/// Monitoring metrics
pub struct MonitorMetrics {
    /// Total jobs submitted
    pub jobs_submitted: Counter,
    /// Total jobs completed
    pub jobs_completed: Counter,
    /// Total jobs failed
    pub jobs_failed: Counter,
    /// Active jobs gauge
    pub jobs_active: Gauge,
    /// Total workers registered
    pub workers_registered: Counter,
    /// Active workers gauge
    pub workers_active: Gauge,
    /// Frame render time histogram
    pub frame_render_time: Histogram,
    /// Queue size gauge
    pub queue_size: Gauge,
}

impl MonitorMetrics {
    /// Create new metrics
    pub fn new(_registry: &Registry) -> Result<Self> {
        Ok(Self {
            jobs_submitted: Counter::new("jobs_submitted_total", "Total jobs submitted")?,
            jobs_completed: Counter::new("jobs_completed_total", "Total jobs completed")?,
            jobs_failed: Counter::new("jobs_failed_total", "Total jobs failed")?,
            jobs_active: Gauge::new("jobs_active", "Active jobs")?,
            workers_registered: Counter::new(
                "workers_registered_total",
                "Total workers registered",
            )?,
            workers_active: Gauge::new("workers_active", "Active workers")?,
            frame_render_time: Histogram::with_opts(prometheus::HistogramOpts::new(
                "frame_render_seconds",
                "Frame render time in seconds",
            ))?,
            queue_size: Gauge::new("queue_size", "Task queue size")?,
        })
    }

    /// Register all metrics
    pub fn register(&self, registry: &Registry) -> Result<()> {
        registry.register(Box::new(self.jobs_submitted.clone()))?;
        registry.register(Box::new(self.jobs_completed.clone()))?;
        registry.register(Box::new(self.jobs_failed.clone()))?;
        registry.register(Box::new(self.jobs_active.clone()))?;
        registry.register(Box::new(self.workers_registered.clone()))?;
        registry.register(Box::new(self.workers_active.clone()))?;
        registry.register(Box::new(self.frame_render_time.clone()))?;
        registry.register(Box::new(self.queue_size.clone()))?;
        Ok(())
    }
}

/// Job progress snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobProgress {
    /// Job ID
    pub job_id: JobId,
    /// Current progress (0.0 to 1.0)
    pub progress: f64,
    /// Frames completed
    pub frames_completed: u32,
    /// Total frames
    pub total_frames: u32,
    /// State
    pub state: JobState,
    /// ETA
    pub eta: Option<DateTime<Utc>>,
}

/// Worker status snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerStatus {
    /// Worker ID
    pub worker_id: WorkerId,
    /// State
    pub state: WorkerState,
    /// Current job
    pub current_job: Option<JobId>,
    /// CPU utilization
    pub cpu_utilization: f64,
    /// Memory utilization
    pub memory_utilization: f64,
    /// Last heartbeat
    pub last_heartbeat: DateTime<Utc>,
}

/// System metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Total jobs
    pub total_jobs: usize,
    /// Active jobs
    pub active_jobs: usize,
    /// Completed jobs
    pub completed_jobs: usize,
    /// Failed jobs
    pub failed_jobs: usize,
    /// Total workers
    pub total_workers: usize,
    /// Idle workers
    pub idle_workers: usize,
    /// Busy workers
    pub busy_workers: usize,
    /// Queue size
    pub queue_size: usize,
    /// Average frame time
    pub avg_frame_time: f64,
    /// Throughput (frames per hour)
    pub throughput: f64,
}

/// Monitor for render farm
pub struct Monitor {
    config: MonitorConfig,
    metrics: Arc<MonitorMetrics>,
    registry: Arc<Registry>,
    job_history: Vec<JobProgress>,
    worker_history: Vec<WorkerStatus>,
    system_history: Vec<SystemMetrics>,
}

impl Monitor {
    /// Create a new monitor
    pub fn new(config: MonitorConfig) -> Result<Self> {
        let registry = Arc::new(Registry::new());
        let metrics = Arc::new(MonitorMetrics::new(&registry)?);
        metrics.register(&registry)?;

        Ok(Self {
            config,
            metrics,
            registry,
            job_history: Vec::new(),
            worker_history: Vec::new(),
            system_history: Vec::new(),
        })
    }

    /// Record job submitted
    pub fn record_job_submitted(&self) {
        if self.config.enable_metrics {
            self.metrics.jobs_submitted.inc();
            self.metrics.jobs_active.inc();
        }
    }

    /// Record job completed
    pub fn record_job_completed(&self) {
        if self.config.enable_metrics {
            self.metrics.jobs_completed.inc();
            self.metrics.jobs_active.dec();
        }
    }

    /// Record job failed
    pub fn record_job_failed(&self) {
        if self.config.enable_metrics {
            self.metrics.jobs_failed.inc();
            self.metrics.jobs_active.dec();
        }
    }

    /// Record worker registered
    pub fn record_worker_registered(&self) {
        if self.config.enable_metrics {
            self.metrics.workers_registered.inc();
            self.metrics.workers_active.inc();
        }
    }

    /// Record worker offline
    pub fn record_worker_offline(&self) {
        if self.config.enable_metrics {
            self.metrics.workers_active.dec();
        }
    }

    /// Record frame render time
    pub fn record_frame_time(&self, seconds: f64) {
        if self.config.enable_metrics {
            self.metrics.frame_render_time.observe(seconds);
        }
    }

    /// Update queue size
    pub fn update_queue_size(&self, size: usize) {
        if self.config.enable_metrics {
            self.metrics.queue_size.set(size as f64);
        }
    }

    /// Capture job progress
    pub fn capture_job_progress(&mut self, progress: JobProgress) {
        self.job_history.push(progress);
        self.cleanup_old_data();
    }

    /// Capture worker status
    pub fn capture_worker_status(&mut self, status: WorkerStatus) {
        self.worker_history.push(status);
        self.cleanup_old_data();
    }

    /// Capture system metrics
    pub fn capture_system_metrics(&mut self, metrics: SystemMetrics) {
        self.system_history.push(metrics);
        self.cleanup_old_data();
    }

    /// Get job progress history
    #[must_use]
    pub fn get_job_progress(&self, job_id: JobId) -> Vec<JobProgress> {
        self.job_history
            .iter()
            .filter(|p| p.job_id == job_id)
            .cloned()
            .collect()
    }

    /// Get worker status history
    #[must_use]
    pub fn get_worker_status(&self, worker_id: WorkerId) -> Vec<WorkerStatus> {
        self.worker_history
            .iter()
            .filter(|s| s.worker_id == worker_id)
            .cloned()
            .collect()
    }

    /// Get latest system metrics
    #[must_use]
    pub fn get_latest_metrics(&self) -> Option<SystemMetrics> {
        self.system_history.last().cloned()
    }

    /// Get metrics history
    #[must_use]
    pub fn get_metrics_history(&self, limit: usize) -> Vec<SystemMetrics> {
        let len = self.system_history.len();
        if len <= limit {
            self.system_history.clone()
        } else {
            self.system_history[len - limit..].to_vec()
        }
    }

    /// Clean up old data
    fn cleanup_old_data(&mut self) {
        let cutoff = Utc::now() - chrono::Duration::hours(i64::from(self.config.retention_hours));

        self.job_history
            .retain(|p| p.eta.map_or(true, |eta| eta > cutoff));

        self.worker_history.retain(|s| s.last_heartbeat > cutoff);

        self.system_history.retain(|m| m.timestamp > cutoff);
    }

    /// Get Prometheus metrics
    #[must_use]
    pub fn get_prometheus_metrics(&self) -> String {
        let encoder = prometheus::TextEncoder::new();
        let metric_families = self.registry.gather();
        encoder
            .encode_to_string(&metric_families)
            .unwrap_or_default()
    }
}

impl Default for Monitor {
    fn default() -> Self {
        // MonitorConfig::default() with a fresh Registry is infallible in practice
        // since all metric names are valid and the registry is empty.
        match Self::new(MonitorConfig::default()) {
            Ok(monitor) => monitor,
            Err(e) => {
                // Disable metrics if prometheus initialisation fails
                let registry = Arc::new(Registry::new());
                // Create metrics without registering them — they still work as counters
                // but won't be exported. Use a disabled config to suppress recording.
                let metrics = Arc::new(MonitorMetrics {
                    jobs_submitted: Counter::new("noop_submitted", "noop")
                        .unwrap_or_else(|_| unreachable!()),
                    jobs_completed: Counter::new("noop_completed", "noop")
                        .unwrap_or_else(|_| unreachable!()),
                    jobs_failed: Counter::new("noop_failed", "noop")
                        .unwrap_or_else(|_| unreachable!()),
                    jobs_active: Gauge::new("noop_active", "noop")
                        .unwrap_or_else(|_| unreachable!()),
                    workers_registered: Counter::new("noop_workers_reg", "noop")
                        .unwrap_or_else(|_| unreachable!()),
                    workers_active: Gauge::new("noop_workers_act", "noop")
                        .unwrap_or_else(|_| unreachable!()),
                    frame_render_time: Histogram::with_opts(prometheus::HistogramOpts::new(
                        "noop_frame_time",
                        "noop",
                    ))
                    .unwrap_or_else(|_| unreachable!()),
                    queue_size: Gauge::new("noop_queue", "noop").unwrap_or_else(|_| unreachable!()),
                });
                eprintln!("warning: Monitor::default() failed ({e}), metrics disabled");
                Self {
                    config: MonitorConfig {
                        enable_metrics: false,
                        ..MonitorConfig::default()
                    },
                    metrics,
                    registry,
                    job_history: Vec::new(),
                    worker_history: Vec::new(),
                    system_history: Vec::new(),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_creation() -> Result<()> {
        let config = MonitorConfig::default();
        let monitor = Monitor::new(config)?;
        assert!(monitor.system_history.is_empty());
        Ok(())
    }

    #[test]
    fn test_record_job_submitted() -> Result<()> {
        let config = MonitorConfig::default();
        let monitor = Monitor::new(config)?;

        monitor.record_job_submitted();
        // Metrics should be updated
        Ok(())
    }

    #[test]
    fn test_record_job_completed() -> Result<()> {
        let config = MonitorConfig::default();
        let monitor = Monitor::new(config)?;

        monitor.record_job_submitted();
        monitor.record_job_completed();
        Ok(())
    }

    #[test]
    fn test_capture_system_metrics() -> Result<()> {
        let config = MonitorConfig::default();
        let mut monitor = Monitor::new(config)?;

        let metrics = SystemMetrics {
            timestamp: Utc::now(),
            total_jobs: 10,
            active_jobs: 5,
            completed_jobs: 4,
            failed_jobs: 1,
            total_workers: 3,
            idle_workers: 1,
            busy_workers: 2,
            queue_size: 20,
            avg_frame_time: 1.5,
            throughput: 100.0,
        };

        monitor.capture_system_metrics(metrics);
        assert_eq!(monitor.system_history.len(), 1);

        let latest = monitor.get_latest_metrics();
        assert!(latest.is_some());
        assert_eq!(latest.expect("should succeed in test").total_jobs, 10);

        Ok(())
    }

    #[test]
    fn test_job_progress_history() -> Result<()> {
        let config = MonitorConfig::default();
        let mut monitor = Monitor::new(config)?;

        let job_id = JobId::new();
        let progress = JobProgress {
            job_id,
            progress: 0.5,
            frames_completed: 50,
            total_frames: 100,
            state: JobState::Rendering,
            eta: Some(Utc::now() + chrono::Duration::hours(1)),
        };

        monitor.capture_job_progress(progress);

        let history = monitor.get_job_progress(job_id);
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].progress, 0.5);

        Ok(())
    }

    #[test]
    fn test_metrics_history_limit() -> Result<()> {
        let config = MonitorConfig::default();
        let mut monitor = Monitor::new(config)?;

        // Add 10 metrics
        for i in 0..10 {
            let metrics = SystemMetrics {
                timestamp: Utc::now(),
                total_jobs: i,
                active_jobs: 0,
                completed_jobs: 0,
                failed_jobs: 0,
                total_workers: 0,
                idle_workers: 0,
                busy_workers: 0,
                queue_size: 0,
                avg_frame_time: 0.0,
                throughput: 0.0,
            };
            monitor.capture_system_metrics(metrics);
        }

        // Get last 5
        let history = monitor.get_metrics_history(5);
        assert_eq!(history.len(), 5);
        assert_eq!(history[0].total_jobs, 5);

        Ok(())
    }

    #[test]
    fn test_prometheus_metrics() -> Result<()> {
        let config = MonitorConfig::default();
        let monitor = Monitor::new(config)?;

        monitor.record_job_submitted();
        monitor.record_job_submitted();
        monitor.record_job_completed();

        let metrics_text = monitor.get_prometheus_metrics();
        assert!(!metrics_text.is_empty());

        Ok(())
    }
}
