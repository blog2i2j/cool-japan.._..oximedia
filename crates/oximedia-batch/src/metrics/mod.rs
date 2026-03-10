//! Prometheus metrics integration

use prometheus::{Counter, Gauge, GaugeVec, HistogramVec, Opts, Registry};
use std::sync::Arc;

/// Metrics collector for batch processing
pub struct MetricsCollector {
    registry: Arc<Registry>,
    jobs_submitted: Counter,
    jobs_completed: Counter,
    jobs_failed: Counter,
    jobs_cancelled: Counter,
    jobs_by_status: GaugeVec,
    jobs_by_priority: GaugeVec,
    job_duration: HistogramVec,
    processing_speed: GaugeVec,
    queue_size: Gauge,
    active_workers: Gauge,
}

impl MetricsCollector {
    /// Create a new metrics collector
    ///
    /// # Errors
    ///
    /// Returns an error if metrics registration fails
    pub fn new() -> Result<Self, prometheus::Error> {
        let registry = Arc::new(Registry::new());

        let jobs_submitted = Counter::with_opts(Opts::new(
            "batch_jobs_submitted_total",
            "Total number of jobs submitted",
        ))?;
        registry.register(Box::new(jobs_submitted.clone()))?;

        let jobs_completed = Counter::with_opts(Opts::new(
            "batch_jobs_completed_total",
            "Total number of jobs completed",
        ))?;
        registry.register(Box::new(jobs_completed.clone()))?;

        let jobs_failed = Counter::with_opts(Opts::new(
            "batch_jobs_failed_total",
            "Total number of jobs failed",
        ))?;
        registry.register(Box::new(jobs_failed.clone()))?;

        let jobs_cancelled = Counter::with_opts(Opts::new(
            "batch_jobs_cancelled_total",
            "Total number of jobs cancelled",
        ))?;
        registry.register(Box::new(jobs_cancelled.clone()))?;

        let jobs_by_status = GaugeVec::new(
            Opts::new("batch_jobs_by_status", "Jobs by status"),
            &["status"],
        )?;
        registry.register(Box::new(jobs_by_status.clone()))?;

        let jobs_by_priority = GaugeVec::new(
            Opts::new("batch_jobs_by_priority", "Jobs by priority"),
            &["priority"],
        )?;
        registry.register(Box::new(jobs_by_priority.clone()))?;

        let job_duration = HistogramVec::new(
            prometheus::HistogramOpts::new("batch_job_duration_seconds", "Job duration in seconds"),
            &["operation"],
        )?;
        registry.register(Box::new(job_duration.clone()))?;

        let processing_speed = GaugeVec::new(
            Opts::new("batch_processing_speed", "Processing speed"),
            &["unit"],
        )?;
        registry.register(Box::new(processing_speed.clone()))?;

        let queue_size =
            Gauge::with_opts(Opts::new("batch_queue_size", "Number of jobs in queue"))?;
        registry.register(Box::new(queue_size.clone()))?;

        let active_workers = Gauge::with_opts(Opts::new(
            "batch_active_workers",
            "Number of active workers",
        ))?;
        registry.register(Box::new(active_workers.clone()))?;

        Ok(Self {
            registry,
            jobs_submitted,
            jobs_completed,
            jobs_failed,
            jobs_cancelled,
            jobs_by_status,
            jobs_by_priority,
            job_duration,
            processing_speed,
            queue_size,
            active_workers,
        })
    }

    /// Increment jobs submitted counter
    pub fn inc_jobs_submitted(&self) {
        self.jobs_submitted.inc();
    }

    /// Increment jobs completed counter
    pub fn inc_jobs_completed(&self) {
        self.jobs_completed.inc();
    }

    /// Increment jobs failed counter
    pub fn inc_jobs_failed(&self) {
        self.jobs_failed.inc();
    }

    /// Increment jobs cancelled counter
    pub fn inc_jobs_cancelled(&self) {
        self.jobs_cancelled.inc();
    }

    /// Set jobs by status
    pub fn set_jobs_by_status(&self, status: &str, count: f64) {
        self.jobs_by_status.with_label_values(&[status]).set(count);
    }

    /// Set jobs by priority
    pub fn set_jobs_by_priority(&self, priority: &str, count: f64) {
        self.jobs_by_priority
            .with_label_values(&[priority])
            .set(count);
    }

    /// Observe job duration
    pub fn observe_job_duration(&self, operation: &str, duration_secs: f64) {
        self.job_duration
            .with_label_values(&[operation])
            .observe(duration_secs);
    }

    /// Set processing speed
    pub fn set_processing_speed(&self, unit: &str, speed: f64) {
        self.processing_speed.with_label_values(&[unit]).set(speed);
    }

    /// Set queue size
    pub fn set_queue_size(&self, size: f64) {
        self.queue_size.set(size);
    }

    /// Set active workers
    pub fn set_active_workers(&self, count: f64) {
        self.active_workers.set(count);
    }

    /// Get registry for HTTP exposure
    #[must_use]
    pub fn registry(&self) -> Arc<Registry> {
        Arc::clone(&self.registry)
    }

    /// Gather metrics in Prometheus format
    #[must_use]
    pub fn gather(&self) -> Vec<prometheus::proto::MetricFamily> {
        self.registry.gather()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_collector_creation() {
        let result = MetricsCollector::new();
        assert!(result.is_ok());
    }

    #[test]
    fn test_inc_jobs_submitted() {
        let collector = MetricsCollector::new().expect("failed to create");
        collector.inc_jobs_submitted();
        collector.inc_jobs_submitted();

        let metrics = collector.gather();
        assert!(!metrics.is_empty());
    }

    #[test]
    fn test_inc_jobs_completed() {
        let collector = MetricsCollector::new().expect("failed to create");
        collector.inc_jobs_completed();

        let metrics = collector.gather();
        assert!(!metrics.is_empty());
    }

    #[test]
    fn test_inc_jobs_failed() {
        let collector = MetricsCollector::new().expect("failed to create");
        collector.inc_jobs_failed();

        let metrics = collector.gather();
        assert!(!metrics.is_empty());
    }

    #[test]
    fn test_set_jobs_by_status() {
        let collector = MetricsCollector::new().expect("failed to create");
        collector.set_jobs_by_status("running", 5.0);

        let metrics = collector.gather();
        assert!(!metrics.is_empty());
    }

    #[test]
    fn test_set_jobs_by_priority() {
        let collector = MetricsCollector::new().expect("failed to create");
        collector.set_jobs_by_priority("high", 3.0);

        let metrics = collector.gather();
        assert!(!metrics.is_empty());
    }

    #[test]
    fn test_observe_job_duration() {
        let collector = MetricsCollector::new().expect("failed to create");
        collector.observe_job_duration("transcode", 120.5);

        let metrics = collector.gather();
        assert!(!metrics.is_empty());
    }

    #[test]
    fn test_set_processing_speed() {
        let collector = MetricsCollector::new().expect("failed to create");
        collector.set_processing_speed("fps", 30.0);

        let metrics = collector.gather();
        assert!(!metrics.is_empty());
    }

    #[test]
    fn test_set_queue_size() {
        let collector = MetricsCollector::new().expect("failed to create");
        collector.set_queue_size(10.0);

        let metrics = collector.gather();
        assert!(!metrics.is_empty());
    }

    #[test]
    fn test_set_active_workers() {
        let collector = MetricsCollector::new().expect("failed to create");
        collector.set_active_workers(4.0);

        let metrics = collector.gather();
        assert!(!metrics.is_empty());
    }
}
