//! Metrics collection and reporting using Prometheus

use prometheus::{Counter, Gauge, GaugeVec, Histogram, HistogramOpts, Opts, Registry};
use std::sync::Arc;

/// Farm metrics collector
pub struct FarmMetrics {
    registry: Arc<Registry>,

    // Job metrics
    jobs_submitted: Counter,
    jobs_completed: Counter,
    jobs_failed: Counter,
    jobs_cancelled: Counter,
    jobs_active: Gauge,
    jobs_queued: Gauge,
    job_duration: Histogram,

    // Task metrics
    tasks_pending: Gauge,
    tasks_running: Gauge,
    tasks_completed: Counter,
    tasks_failed: Counter,
    task_duration: Histogram,
    task_retries: Counter,

    // Worker metrics
    workers_registered: Gauge,
    workers_active: Gauge,
    workers_idle: Gauge,
    workers_busy: Gauge,
    workers_offline: Gauge,

    // Queue metrics
    queue_depth: Gauge,
    queue_depth_by_priority: GaugeVec,
    queue_wait_time: Histogram,

    // Resource metrics
    cpu_utilization: GaugeVec,
    memory_utilization: GaugeVec,
    disk_utilization: GaugeVec,

    // Throughput metrics
    bytes_processed: Counter,
    frames_processed: Counter,
}

impl FarmMetrics {
    /// Create a new metrics collector
    pub fn new() -> Result<Self, prometheus::Error> {
        let registry = Arc::new(Registry::new());

        // Job metrics
        let jobs_submitted = Counter::with_opts(Opts::new(
            "farm_jobs_submitted_total",
            "Total number of jobs submitted",
        ))?;
        registry.register(Box::new(jobs_submitted.clone()))?;

        let jobs_completed = Counter::with_opts(Opts::new(
            "farm_jobs_completed_total",
            "Total number of jobs completed",
        ))?;
        registry.register(Box::new(jobs_completed.clone()))?;

        let jobs_failed = Counter::with_opts(Opts::new(
            "farm_jobs_failed_total",
            "Total number of jobs failed",
        ))?;
        registry.register(Box::new(jobs_failed.clone()))?;

        let jobs_cancelled = Counter::with_opts(Opts::new(
            "farm_jobs_cancelled_total",
            "Total number of jobs cancelled",
        ))?;
        registry.register(Box::new(jobs_cancelled.clone()))?;

        let jobs_active = Gauge::with_opts(Opts::new(
            "farm_jobs_active",
            "Number of currently active jobs",
        ))?;
        registry.register(Box::new(jobs_active.clone()))?;

        let jobs_queued = Gauge::with_opts(Opts::new(
            "farm_jobs_queued",
            "Number of currently queued jobs",
        ))?;
        registry.register(Box::new(jobs_queued.clone()))?;

        let job_duration = Histogram::with_opts(
            HistogramOpts::new("farm_job_duration_seconds", "Job duration in seconds")
                .buckets(vec![1.0, 10.0, 60.0, 300.0, 600.0, 1800.0, 3600.0]),
        )?;
        registry.register(Box::new(job_duration.clone()))?;

        // Task metrics
        let tasks_pending =
            Gauge::with_opts(Opts::new("farm_tasks_pending", "Number of pending tasks"))?;
        registry.register(Box::new(tasks_pending.clone()))?;

        let tasks_running =
            Gauge::with_opts(Opts::new("farm_tasks_running", "Number of running tasks"))?;
        registry.register(Box::new(tasks_running.clone()))?;

        let tasks_completed = Counter::with_opts(Opts::new(
            "farm_tasks_completed_total",
            "Total number of tasks completed",
        ))?;
        registry.register(Box::new(tasks_completed.clone()))?;

        let tasks_failed = Counter::with_opts(Opts::new(
            "farm_tasks_failed_total",
            "Total number of tasks failed",
        ))?;
        registry.register(Box::new(tasks_failed.clone()))?;

        let task_duration = Histogram::with_opts(
            HistogramOpts::new("farm_task_duration_seconds", "Task duration in seconds")
                .buckets(vec![0.1, 1.0, 10.0, 60.0, 300.0, 600.0]),
        )?;
        registry.register(Box::new(task_duration.clone()))?;

        let task_retries = Counter::with_opts(Opts::new(
            "farm_task_retries_total",
            "Total number of task retries",
        ))?;
        registry.register(Box::new(task_retries.clone()))?;

        // Worker metrics
        let workers_registered = Gauge::with_opts(Opts::new(
            "farm_workers_registered",
            "Number of registered workers",
        ))?;
        registry.register(Box::new(workers_registered.clone()))?;

        let workers_active =
            Gauge::with_opts(Opts::new("farm_workers_active", "Number of active workers"))?;
        registry.register(Box::new(workers_active.clone()))?;

        let workers_idle =
            Gauge::with_opts(Opts::new("farm_workers_idle", "Number of idle workers"))?;
        registry.register(Box::new(workers_idle.clone()))?;

        let workers_busy =
            Gauge::with_opts(Opts::new("farm_workers_busy", "Number of busy workers"))?;
        registry.register(Box::new(workers_busy.clone()))?;

        let workers_offline = Gauge::with_opts(Opts::new(
            "farm_workers_offline",
            "Number of offline workers",
        ))?;
        registry.register(Box::new(workers_offline.clone()))?;

        // Queue metrics
        let queue_depth = Gauge::with_opts(Opts::new("farm_queue_depth", "Total queue depth"))?;
        registry.register(Box::new(queue_depth.clone()))?;

        let queue_depth_by_priority = GaugeVec::new(
            Opts::new("farm_queue_depth_by_priority", "Queue depth by priority"),
            &["priority"],
        )?;
        registry.register(Box::new(queue_depth_by_priority.clone()))?;

        let queue_wait_time = Histogram::with_opts(
            HistogramOpts::new("farm_queue_wait_time_seconds", "Queue wait time in seconds")
                .buckets(vec![1.0, 5.0, 10.0, 30.0, 60.0, 300.0]),
        )?;
        registry.register(Box::new(queue_wait_time.clone()))?;

        // Resource metrics
        let cpu_utilization = GaugeVec::new(
            Opts::new("farm_cpu_utilization", "CPU utilization"),
            &["worker"],
        )?;
        registry.register(Box::new(cpu_utilization.clone()))?;

        let memory_utilization = GaugeVec::new(
            Opts::new("farm_memory_utilization", "Memory utilization"),
            &["worker"],
        )?;
        registry.register(Box::new(memory_utilization.clone()))?;

        let disk_utilization = GaugeVec::new(
            Opts::new("farm_disk_utilization", "Disk utilization"),
            &["worker"],
        )?;
        registry.register(Box::new(disk_utilization.clone()))?;

        // Throughput metrics
        let bytes_processed = Counter::with_opts(Opts::new(
            "farm_bytes_processed_total",
            "Total bytes processed",
        ))?;
        registry.register(Box::new(bytes_processed.clone()))?;

        let frames_processed = Counter::with_opts(Opts::new(
            "farm_frames_processed_total",
            "Total frames processed",
        ))?;
        registry.register(Box::new(frames_processed.clone()))?;

        Ok(Self {
            registry,
            jobs_submitted,
            jobs_completed,
            jobs_failed,
            jobs_cancelled,
            jobs_active,
            jobs_queued,
            job_duration,
            tasks_pending,
            tasks_running,
            tasks_completed,
            tasks_failed,
            task_duration,
            task_retries,
            workers_registered,
            workers_active,
            workers_idle,
            workers_busy,
            workers_offline,
            queue_depth,
            queue_depth_by_priority,
            queue_wait_time,
            cpu_utilization,
            memory_utilization,
            disk_utilization,
            bytes_processed,
            frames_processed,
        })
    }

    /// Record job submission
    pub fn record_job_submitted(&self) {
        self.jobs_submitted.inc();
    }

    /// Record job completion
    pub fn record_job_completed(&self, duration_secs: f64) {
        self.jobs_completed.inc();
        self.job_duration.observe(duration_secs);
    }

    /// Record job failure
    pub fn record_job_failed(&self) {
        self.jobs_failed.inc();
    }

    /// Record job cancellation
    pub fn record_job_cancelled(&self) {
        self.jobs_cancelled.inc();
    }

    /// Set active jobs count
    pub fn set_jobs_active(&self, count: usize) {
        self.jobs_active.set(count as f64);
    }

    /// Set queued jobs count
    pub fn set_jobs_queued(&self, count: usize) {
        self.jobs_queued.set(count as f64);
    }

    /// Record task completion
    pub fn record_task_completed(&self, duration_secs: f64) {
        self.tasks_completed.inc();
        self.task_duration.observe(duration_secs);
    }

    /// Record task failure
    pub fn record_task_failed(&self) {
        self.tasks_failed.inc();
    }

    /// Record task retry
    pub fn record_task_retry(&self) {
        self.task_retries.inc();
    }

    /// Set pending tasks count
    pub fn set_tasks_pending(&self, count: usize) {
        self.tasks_pending.set(count as f64);
    }

    /// Set running tasks count
    pub fn set_tasks_running(&self, count: usize) {
        self.tasks_running.set(count as f64);
    }

    /// Update worker counts
    pub fn update_worker_counts(
        &self,
        registered: usize,
        active: usize,
        idle: usize,
        busy: usize,
        offline: usize,
    ) {
        self.workers_registered.set(registered as f64);
        self.workers_active.set(active as f64);
        self.workers_idle.set(idle as f64);
        self.workers_busy.set(busy as f64);
        self.workers_offline.set(offline as f64);
    }

    /// Set queue depth
    pub fn set_queue_depth(&self, depth: usize) {
        self.queue_depth.set(depth as f64);
    }

    /// Set queue depth by priority
    pub fn set_queue_depth_by_priority(&self, priority: &str, depth: usize) {
        self.queue_depth_by_priority
            .with_label_values(&[priority])
            .set(depth as f64);
    }

    /// Record queue wait time
    pub fn record_queue_wait_time(&self, wait_time_secs: f64) {
        self.queue_wait_time.observe(wait_time_secs);
    }

    /// Update worker resource utilization
    pub fn update_worker_resources(&self, worker_id: &str, cpu: f64, memory: f64, disk: f64) {
        self.cpu_utilization
            .with_label_values(&[worker_id])
            .set(cpu);
        self.memory_utilization
            .with_label_values(&[worker_id])
            .set(memory);
        self.disk_utilization
            .with_label_values(&[worker_id])
            .set(disk);
    }

    /// Record bytes processed
    pub fn record_bytes_processed(&self, bytes: u64) {
        self.bytes_processed.inc_by(bytes as f64);
    }

    /// Record frames processed
    pub fn record_frames_processed(&self, frames: u64) {
        self.frames_processed.inc_by(frames as f64);
    }

    /// Get the Prometheus registry
    #[must_use]
    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    /// Gather metrics in Prometheus text format
    #[must_use]
    pub fn gather(&self) -> String {
        use prometheus::Encoder;
        let encoder = prometheus::TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        if encoder.encode(&metric_families, &mut buffer).is_err() {
            return String::new();
        }
        String::from_utf8(buffer).unwrap_or_default()
    }
}

impl Default for FarmMetrics {
    fn default() -> Self {
        Self::new().expect("Failed to create farm metrics")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let metrics = FarmMetrics::new().unwrap();
        let output = metrics.gather();
        assert!(!output.is_empty());
    }

    #[test]
    fn test_job_metrics() {
        let metrics = FarmMetrics::new().unwrap();

        metrics.record_job_submitted();
        metrics.record_job_completed(10.0);
        metrics.record_job_failed();
        metrics.record_job_cancelled();
        metrics.set_jobs_active(5);
        metrics.set_jobs_queued(10);

        let output = metrics.gather();
        assert!(output.contains("farm_jobs_submitted_total"));
        assert!(output.contains("farm_jobs_completed_total"));
        assert!(output.contains("farm_jobs_failed_total"));
    }

    #[test]
    fn test_task_metrics() {
        let metrics = FarmMetrics::new().unwrap();

        metrics.record_task_completed(5.0);
        metrics.record_task_failed();
        metrics.record_task_retry();
        metrics.set_tasks_pending(3);
        metrics.set_tasks_running(2);

        let output = metrics.gather();
        assert!(output.contains("farm_tasks_completed_total"));
        assert!(output.contains("farm_tasks_failed_total"));
    }

    #[test]
    fn test_worker_metrics() {
        let metrics = FarmMetrics::new().unwrap();

        metrics.update_worker_counts(10, 8, 3, 4, 1);

        let output = metrics.gather();
        assert!(output.contains("farm_workers_registered"));
        assert!(output.contains("farm_workers_active"));
    }

    #[test]
    fn test_queue_metrics() {
        let metrics = FarmMetrics::new().unwrap();

        metrics.set_queue_depth(15);
        metrics.set_queue_depth_by_priority("high", 5);
        metrics.set_queue_depth_by_priority("normal", 10);
        metrics.record_queue_wait_time(3.5);

        let output = metrics.gather();
        assert!(output.contains("farm_queue_depth"));
    }

    #[test]
    fn test_resource_metrics() {
        let metrics = FarmMetrics::new().unwrap();

        metrics.update_worker_resources("worker-1", 0.75, 0.60, 0.50);

        let output = metrics.gather();
        assert!(output.contains("farm_cpu_utilization"));
        assert!(output.contains("farm_memory_utilization"));
        assert!(output.contains("farm_disk_utilization"));
    }

    #[test]
    fn test_throughput_metrics() {
        let metrics = FarmMetrics::new().unwrap();

        metrics.record_bytes_processed(1024 * 1024 * 100); // 100MB
        metrics.record_frames_processed(3000);

        let output = metrics.gather();
        assert!(output.contains("farm_bytes_processed_total"));
        assert!(output.contains("farm_frames_processed_total"));
    }
}
