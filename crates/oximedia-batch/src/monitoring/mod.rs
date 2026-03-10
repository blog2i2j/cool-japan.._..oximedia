//! Monitoring and progress tracking

pub mod progress;
pub mod reporter;
pub mod stats;

use crate::types::JobId;
use dashmap::DashMap;
use progress::ProgressTracker;
use std::sync::Arc;

/// Job progress information
#[derive(Debug, Clone)]
pub struct JobProgress {
    /// Job ID
    pub job_id: JobId,
    /// Progress percentage (0-100)
    pub progress: f64,
    /// Estimated time remaining in seconds
    pub estimated_remaining_secs: Option<u64>,
    /// Processing speed (e.g., FPS, MB/s)
    pub processing_speed: Option<f64>,
    /// Speed unit
    pub speed_unit: String,
    /// Current stage description
    pub stage: String,
}

impl JobProgress {
    /// Create a new job progress
    #[must_use]
    pub fn new(job_id: JobId) -> Self {
        Self {
            job_id,
            progress: 0.0,
            estimated_remaining_secs: None,
            processing_speed: None,
            speed_unit: String::new(),
            stage: "Initializing".to_string(),
        }
    }

    /// Update progress
    pub fn update(&mut self, progress: f64) {
        self.progress = progress.clamp(0.0, 100.0);
    }

    /// Set estimated remaining time
    pub fn set_estimated_remaining(&mut self, secs: u64) {
        self.estimated_remaining_secs = Some(secs);
    }

    /// Set processing speed
    pub fn set_processing_speed(&mut self, speed: f64, unit: String) {
        self.processing_speed = Some(speed);
        self.speed_unit = unit;
    }

    /// Set current stage
    pub fn set_stage(&mut self, stage: String) {
        self.stage = stage;
    }

    /// Check if completed
    #[must_use]
    pub fn is_completed(&self) -> bool {
        self.progress >= 100.0
    }
}

/// Progress monitor
pub struct ProgressMonitor {
    trackers: Arc<DashMap<JobId, ProgressTracker>>,
}

impl ProgressMonitor {
    /// Create a new progress monitor
    #[must_use]
    pub fn new() -> Self {
        Self {
            trackers: Arc::new(DashMap::new()),
        }
    }

    /// Start tracking a job
    pub fn start_tracking(&self, job_id: JobId, total_items: u64) {
        let tracker = ProgressTracker::new(total_items);
        self.trackers.insert(job_id, tracker);
    }

    /// Update progress for a job
    ///
    /// # Arguments
    ///
    /// * `job_id` - ID of the job
    /// * `completed_items` - Number of completed items
    pub fn update_progress(&self, job_id: &JobId, completed_items: u64) {
        if let Some(mut tracker) = self.trackers.get_mut(job_id) {
            tracker.update(completed_items);
        }
    }

    /// Get progress for a job
    ///
    /// # Arguments
    ///
    /// * `job_id` - ID of the job
    #[must_use]
    pub fn get_progress(&self, job_id: &JobId) -> Option<JobProgress> {
        self.trackers.get(job_id).map(|tracker| {
            let mut progress = JobProgress::new(job_id.clone());
            progress.update(tracker.progress_percentage());

            if let Some(eta) = tracker.estimated_remaining_secs() {
                progress.set_estimated_remaining(eta);
            }

            progress
        })
    }

    /// Stop tracking a job
    pub fn stop_tracking(&self, job_id: &JobId) {
        self.trackers.remove(job_id);
    }

    /// Get all active job progresses
    #[must_use]
    pub fn get_all_progress(&self) -> Vec<JobProgress> {
        self.trackers
            .iter()
            .map(|entry| {
                let job_id = entry.key().clone();
                let tracker = entry.value();

                let mut progress = JobProgress::new(job_id);
                progress.update(tracker.progress_percentage());

                if let Some(eta) = tracker.estimated_remaining_secs() {
                    progress.set_estimated_remaining(eta);
                }

                progress
            })
            .collect()
    }
}

impl Default for ProgressMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_progress_creation() {
        let job_id = JobId::new();
        let progress = JobProgress::new(job_id);

        assert_eq!(progress.progress, 0.0);
        assert!(!progress.is_completed());
    }

    #[test]
    fn test_job_progress_update() {
        let job_id = JobId::new();
        let mut progress = JobProgress::new(job_id);

        progress.update(50.0);
        assert_eq!(progress.progress, 50.0);

        progress.update(100.0);
        assert_eq!(progress.progress, 100.0);
        assert!(progress.is_completed());
    }

    #[test]
    fn test_job_progress_clamping() {
        let job_id = JobId::new();
        let mut progress = JobProgress::new(job_id);

        progress.update(150.0);
        assert_eq!(progress.progress, 100.0);

        progress.update(-10.0);
        assert_eq!(progress.progress, 0.0);
    }

    #[test]
    fn test_progress_monitor_creation() {
        let monitor = ProgressMonitor::new();
        assert_eq!(monitor.trackers.len(), 0);
    }

    #[test]
    fn test_start_tracking() {
        let monitor = ProgressMonitor::new();
        let job_id = JobId::new();

        monitor.start_tracking(job_id.clone(), 100);
        assert_eq!(monitor.trackers.len(), 1);
    }

    #[test]
    fn test_update_progress() {
        let monitor = ProgressMonitor::new();
        let job_id = JobId::new();

        monitor.start_tracking(job_id.clone(), 100);
        monitor.update_progress(&job_id, 50);

        let progress = monitor.get_progress(&job_id);
        assert!(progress.is_some());
        assert_eq!(progress.expect("progress should be valid").progress, 50.0);
    }

    #[test]
    fn test_stop_tracking() {
        let monitor = ProgressMonitor::new();
        let job_id = JobId::new();

        monitor.start_tracking(job_id.clone(), 100);
        monitor.stop_tracking(&job_id);

        assert_eq!(monitor.trackers.len(), 0);
    }

    #[test]
    fn test_get_all_progress() {
        let monitor = ProgressMonitor::new();

        let job1 = JobId::new();
        let job2 = JobId::new();

        monitor.start_tracking(job1, 100);
        monitor.start_tracking(job2, 200);

        let all_progress = monitor.get_all_progress();
        assert_eq!(all_progress.len(), 2);
    }
}
