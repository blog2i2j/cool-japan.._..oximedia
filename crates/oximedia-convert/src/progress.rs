/// Conversion progress tracking for `OxiMedia`.
///
/// Provides per-job and multi-job progress tracking with ETA estimation
/// based on observed frame-rate.
///
/// Tracks the progress of a single conversion job.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ConversionProgress {
    /// Unique job identifier.
    pub job_id: u64,
    /// Total number of frames to encode.
    pub frames_total: u64,
    /// Number of frames encoded so far.
    pub frames_done: u64,
    /// Total bytes written to the output so far.
    pub bytes_written: u64,
    /// Wall-clock milliseconds elapsed since the job started.
    pub elapsed_ms: u64,
    /// Observed encoding speed in frames per second.
    pub fps: f64,
    /// Estimated milliseconds remaining, or `None` if not yet computable.
    pub eta_ms: Option<u64>,
}

impl ConversionProgress {
    /// Creates a new progress tracker for a job.
    ///
    /// * `job_id`       – unique identifier for the job
    /// * `total_frames` – total frames expected to be encoded
    #[allow(dead_code)]
    #[must_use]
    pub fn new(job_id: u64, total_frames: u64) -> Self {
        Self {
            job_id,
            frames_total: total_frames,
            frames_done: 0,
            bytes_written: 0,
            elapsed_ms: 0,
            fps: 0.0,
            eta_ms: None,
        }
    }

    /// Updates the progress with the latest counts.
    ///
    /// * `frames_done` – cumulative number of frames encoded
    /// * `bytes`       – cumulative bytes written
    /// * `elapsed_ms`  – wall-clock milliseconds since job start
    #[allow(dead_code)]
    pub fn update(&mut self, frames_done: u64, bytes: u64, elapsed_ms: u64) {
        self.frames_done = frames_done;
        self.bytes_written = bytes;
        self.elapsed_ms = elapsed_ms;

        // Recompute FPS
        if elapsed_ms > 0 {
            self.fps = frames_done as f64 / (elapsed_ms as f64 / 1_000.0);
        } else {
            self.fps = 0.0;
        }

        // Recompute ETA
        if self.fps > 0.0 && frames_done < self.frames_total {
            let remaining_frames = self.frames_total - frames_done;
            let remaining_s = remaining_frames as f64 / self.fps;
            self.eta_ms = Some((remaining_s * 1_000.0) as u64);
        } else if frames_done >= self.frames_total {
            self.eta_ms = Some(0);
        } else {
            self.eta_ms = None;
        }
    }

    /// Returns the completion percentage in `[0.0, 100.0]`.
    ///
    /// Returns `100.0` if `frames_total` is zero (nothing to encode).
    #[allow(dead_code)]
    #[must_use]
    pub fn pct(&self) -> f64 {
        if self.frames_total == 0 {
            return 100.0;
        }
        (self.frames_done as f64 / self.frames_total as f64 * 100.0).min(100.0)
    }

    /// Returns the current encoding speed in frames per second.
    #[allow(dead_code)]
    #[must_use]
    pub fn fps(&self) -> f64 {
        self.fps
    }

    /// Returns the estimated remaining time in seconds, or `None`.
    #[allow(dead_code)]
    #[must_use]
    pub fn eta_s(&self) -> Option<f64> {
        self.eta_ms.map(|ms| ms as f64 / 1_000.0)
    }

    /// Returns `true` if the job is complete.
    #[allow(dead_code)]
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.frames_done >= self.frames_total
    }

    /// Returns frames remaining.
    #[allow(dead_code)]
    #[must_use]
    pub fn frames_remaining(&self) -> u64 {
        self.frames_total.saturating_sub(self.frames_done)
    }

    /// Returns elapsed time in seconds.
    #[allow(dead_code)]
    #[must_use]
    pub fn elapsed_s(&self) -> f64 {
        self.elapsed_ms as f64 / 1_000.0
    }
}

// ── MultiJobProgress ──────────────────────────────────────────────────────────

/// Aggregates progress across multiple conversion jobs.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct MultiJobProgress {
    /// Individual job progress entries.
    jobs: Vec<ConversionProgress>,
}

impl MultiJobProgress {
    /// Creates a new empty multi-job tracker.
    #[allow(dead_code)]
    #[must_use]
    pub fn new() -> Self {
        Self { jobs: Vec::new() }
    }

    /// Adds a job progress record.
    #[allow(dead_code)]
    pub fn add_job(&mut self, p: ConversionProgress) {
        self.jobs.push(p);
    }

    /// Returns the overall completion percentage across all jobs.
    ///
    /// Computed as the weighted average by `frames_total`.
    /// Returns `100.0` if there are no jobs (vacuously complete).
    #[allow(dead_code)]
    #[must_use]
    pub fn overall_pct(&self) -> f64 {
        let total_frames: u64 = self.jobs.iter().map(|j| j.frames_total).sum();
        if total_frames == 0 {
            return 100.0;
        }
        let done_frames: u64 = self.jobs.iter().map(|j| j.frames_done).sum();
        (done_frames as f64 / total_frames as f64 * 100.0).min(100.0)
    }

    /// Returns the number of completed jobs.
    #[allow(dead_code)]
    #[must_use]
    pub fn completed_count(&self) -> usize {
        self.jobs.iter().filter(|j| j.is_complete()).count()
    }

    /// Returns the number of jobs that have started but are not yet complete.
    #[allow(dead_code)]
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.jobs
            .iter()
            .filter(|j| j.frames_done > 0 && !j.is_complete())
            .count()
    }

    /// Returns the total number of jobs (complete + active + pending).
    #[allow(dead_code)]
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.jobs.len()
    }

    /// Returns the number of pending jobs (not yet started).
    #[allow(dead_code)]
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.jobs
            .iter()
            .filter(|j| j.frames_done == 0 && !j.is_complete())
            .count()
    }

    /// Returns a reference to the progress of a job by its `job_id`, if present.
    #[allow(dead_code)]
    #[must_use]
    pub fn find(&self, job_id: u64) -> Option<&ConversionProgress> {
        self.jobs.iter().find(|j| j.job_id == job_id)
    }

    /// Returns the combined bytes written across all jobs.
    #[allow(dead_code)]
    #[must_use]
    pub fn total_bytes_written(&self) -> u64 {
        self.jobs.iter().map(|j| j.bytes_written).sum()
    }

    /// Returns the average FPS across jobs that are currently encoding.
    ///
    /// Returns `0.0` if no jobs are active.
    #[allow(dead_code)]
    #[must_use]
    pub fn average_fps(&self) -> f64 {
        let active: Vec<f64> = self
            .jobs
            .iter()
            .filter(|j| j.fps > 0.0 && !j.is_complete())
            .map(|j| j.fps)
            .collect();
        if active.is_empty() {
            0.0
        } else {
            active.iter().sum::<f64>() / active.len() as f64
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // ── ConversionProgress ────────────────────────────────────────────────────

    #[test]
    fn progress_new_starts_at_zero() {
        let p = ConversionProgress::new(1, 1_000);
        assert_eq!(p.frames_done, 0);
        assert_eq!(p.pct(), 0.0);
        assert!(!p.is_complete());
    }

    #[test]
    fn progress_pct_at_halfway() {
        let mut p = ConversionProgress::new(1, 1_000);
        p.update(500, 1_024 * 1_024, 5_000);
        assert!((p.pct() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn progress_pct_at_complete() {
        let mut p = ConversionProgress::new(1, 100);
        p.update(100, 0, 1_000);
        assert_eq!(p.pct(), 100.0);
        assert!(p.is_complete());
    }

    #[test]
    fn progress_fps_computed_correctly() {
        let mut p = ConversionProgress::new(1, 1_000);
        // 500 frames in 2000 ms = 250 fps
        p.update(500, 0, 2_000);
        assert!((p.fps() - 250.0).abs() < 1e-6);
    }

    #[test]
    fn progress_eta_computed() {
        let mut p = ConversionProgress::new(1, 1_000);
        // 500 frames in 2000 ms → fps = 250 → 500 remaining / 250 = 2s
        p.update(500, 0, 2_000);
        let eta = p.eta_s().expect("ETA should be computed after update");
        assert!((eta - 2.0).abs() < 1e-6);
    }

    #[test]
    fn progress_eta_none_when_fps_zero() {
        let p = ConversionProgress::new(1, 1_000);
        assert!(p.eta_s().is_none());
    }

    #[test]
    fn progress_eta_zero_when_complete() {
        let mut p = ConversionProgress::new(1, 100);
        p.update(100, 0, 500);
        assert_eq!(p.eta_s(), Some(0.0));
    }

    #[test]
    fn progress_frames_remaining() {
        let mut p = ConversionProgress::new(1, 1_000);
        p.update(300, 0, 1_000);
        assert_eq!(p.frames_remaining(), 700);
    }

    #[test]
    fn progress_elapsed_s() {
        let mut p = ConversionProgress::new(1, 1_000);
        p.update(100, 0, 3_500);
        assert!((p.elapsed_s() - 3.5).abs() < 1e-9);
    }

    #[test]
    fn progress_zero_total_frames_is_complete() {
        let p = ConversionProgress::new(42, 0);
        assert_eq!(p.pct(), 100.0);
    }

    // ── MultiJobProgress ──────────────────────────────────────────────────────

    #[test]
    fn multi_empty_is_100pct() {
        let m = MultiJobProgress::new();
        assert_eq!(m.overall_pct(), 100.0);
    }

    #[test]
    fn multi_overall_pct_weighted() {
        let mut m = MultiJobProgress::new();
        // Job A: 500/1000 frames done
        let mut a = ConversionProgress::new(1, 1_000);
        a.update(500, 0, 1_000);
        // Job B: 1000/1000 frames done
        let mut b = ConversionProgress::new(2, 1_000);
        b.update(1_000, 0, 2_000);
        m.add_job(a);
        m.add_job(b);
        // Overall = 1500/2000 = 75%
        assert!((m.overall_pct() - 75.0).abs() < 1e-9);
    }

    #[test]
    fn multi_completed_count() {
        let mut m = MultiJobProgress::new();
        let mut done = ConversionProgress::new(1, 100);
        done.update(100, 0, 1_000);
        let not_done = ConversionProgress::new(2, 100);
        m.add_job(done);
        m.add_job(not_done);
        assert_eq!(m.completed_count(), 1);
    }

    #[test]
    fn multi_active_count() {
        let mut m = MultiJobProgress::new();
        let mut active = ConversionProgress::new(1, 100);
        active.update(50, 0, 500);
        let pending = ConversionProgress::new(2, 100);
        m.add_job(active);
        m.add_job(pending);
        assert_eq!(m.active_count(), 1);
    }

    #[test]
    fn multi_find_by_job_id() {
        let mut m = MultiJobProgress::new();
        m.add_job(ConversionProgress::new(7, 100));
        assert!(m.find(7).is_some());
        assert!(m.find(99).is_none());
    }

    #[test]
    fn multi_total_bytes_written() {
        let mut m = MultiJobProgress::new();
        let mut a = ConversionProgress::new(1, 100);
        a.update(50, 1_000, 500);
        let mut b = ConversionProgress::new(2, 100);
        b.update(50, 2_000, 500);
        m.add_job(a);
        m.add_job(b);
        assert_eq!(m.total_bytes_written(), 3_000);
    }
}
