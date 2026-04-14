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

    /// Returns the overall ETA in seconds across all active jobs.
    ///
    /// Derived from remaining frames and the average FPS of active jobs.
    /// Returns `None` if no active jobs have a meaningful FPS.
    #[allow(dead_code)]
    #[must_use]
    pub fn overall_eta_s(&self) -> Option<f64> {
        let avg_fps = self.average_fps();
        if avg_fps <= 0.0 {
            return None;
        }
        let remaining: u64 = self
            .jobs
            .iter()
            .filter(|j| !j.is_complete())
            .map(|j| j.frames_remaining())
            .sum();
        if remaining == 0 {
            return Some(0.0);
        }
        Some(remaining as f64 / avg_fps)
    }
}

// ── EwmaProgress ──────────────────────────────────────────────────────────────

/// Exponentially-weighted moving average (EWMA) ETA tracker.
///
/// Uses a configurable smoothing factor `alpha ∈ (0, 1]` to blend the
/// instantaneous encoding speed into a running estimate.  A lower alpha
/// produces a smoother, slower-to-react ETA; a higher alpha reacts quickly.
///
/// # Algorithm
///
/// Each call to [`EwmaProgress::record_sample`] provides the frames processed
/// and elapsed wall-clock time for that segment.  The instantaneous FPS is
/// blended into the EWMA:
///
/// ```text
/// ewma_fps = alpha × instant_fps + (1 − alpha) × ewma_fps
/// ```
///
/// ETA is then:
/// ```text
/// eta_s = frames_remaining / ewma_fps
/// ```
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct EwmaProgress {
    /// Total number of frames in the job.
    pub frames_total: u64,
    /// Number of frames processed so far.
    pub frames_done: u64,
    /// Current EWMA FPS estimate.
    ewma_fps: f64,
    /// Smoothing factor α ∈ (0, 1].
    alpha: f64,
    /// Ring-buffer of `(frames_in_segment, elapsed_ms)` samples.
    history: Vec<(u64, u64)>,
    /// Write index into `history`.
    history_idx: usize,
    /// Number of valid entries in `history`.
    history_len: usize,
}

impl EwmaProgress {
    /// Maximum number of segment samples to keep in the sliding window.
    pub const MAX_HISTORY: usize = 16;

    /// Default smoothing factor (α = 0.3).
    pub const DEFAULT_ALPHA: f64 = 0.3;

    /// Create a new EWMA progress tracker.
    ///
    /// * `frames_total` – total frames expected (must be > 0).
    /// * `alpha`        – EWMA smoothing factor ∈ (0, 1]; clamped.
    #[allow(dead_code)]
    #[must_use]
    pub fn new(frames_total: u64, alpha: f64) -> Self {
        let alpha = alpha.clamp(f64::EPSILON, 1.0);
        Self {
            frames_total: frames_total.max(1),
            frames_done: 0,
            ewma_fps: 0.0,
            alpha,
            history: vec![(0, 0); Self::MAX_HISTORY],
            history_idx: 0,
            history_len: 0,
        }
    }

    /// Create a tracker with the default α = 0.3.
    #[allow(dead_code)]
    #[must_use]
    pub fn with_default_alpha(frames_total: u64) -> Self {
        Self::new(frames_total, Self::DEFAULT_ALPHA)
    }

    /// Record a completed segment.
    ///
    /// * `frames_in_segment` – frames encoded in this segment (must be > 0).
    /// * `elapsed_ms`        – wall-clock milliseconds taken for this segment.
    #[allow(dead_code)]
    pub fn record_sample(&mut self, frames_in_segment: u64, elapsed_ms: u64) {
        if frames_in_segment == 0 || elapsed_ms == 0 {
            return;
        }
        self.frames_done = (self.frames_done + frames_in_segment).min(self.frames_total);

        let instant_fps = frames_in_segment as f64 / (elapsed_ms as f64 / 1_000.0);
        if self.ewma_fps <= 0.0 {
            self.ewma_fps = instant_fps;
        } else {
            self.ewma_fps = self.alpha * instant_fps + (1.0 - self.alpha) * self.ewma_fps;
        }

        self.history[self.history_idx] = (frames_in_segment, elapsed_ms);
        self.history_idx = (self.history_idx + 1) % Self::MAX_HISTORY;
        self.history_len = (self.history_len + 1).min(Self::MAX_HISTORY);
    }

    /// Returns the current EWMA FPS estimate.
    #[allow(dead_code)]
    #[must_use]
    pub fn ewma_fps(&self) -> f64 {
        self.ewma_fps
    }

    /// Windowed-average FPS computed from the recent history buffer.
    #[allow(dead_code)]
    #[must_use]
    pub fn windowed_fps(&self) -> f64 {
        if self.history_len == 0 {
            return 0.0;
        }
        let total_frames: u64 = self.history[..self.history_len]
            .iter()
            .map(|(f, _)| f)
            .sum();
        let total_ms: u64 = self.history[..self.history_len]
            .iter()
            .map(|(_, ms)| ms)
            .sum();
        if total_ms == 0 {
            return 0.0;
        }
        total_frames as f64 / (total_ms as f64 / 1_000.0)
    }

    /// EWMA-based ETA in seconds.
    ///
    /// Returns `None` until the first sample has been recorded.
    /// Returns `Some(0.0)` when all frames are done.
    #[allow(dead_code)]
    #[must_use]
    pub fn eta_ewma_s(&self) -> Option<f64> {
        if self.ewma_fps <= 0.0 {
            return None;
        }
        let remaining = self.frames_total.saturating_sub(self.frames_done);
        if remaining == 0 {
            return Some(0.0);
        }
        Some(remaining as f64 / self.ewma_fps)
    }

    /// Windowed-average-based ETA in seconds.
    #[allow(dead_code)]
    #[must_use]
    pub fn eta_windowed_s(&self) -> Option<f64> {
        let wfps = self.windowed_fps();
        if wfps <= 0.0 {
            return None;
        }
        let remaining = self.frames_total.saturating_sub(self.frames_done);
        if remaining == 0 {
            return Some(0.0);
        }
        Some(remaining as f64 / wfps)
    }

    /// Blended ETA: weighted average of EWMA (weight `w_ewma`) and windowed.
    ///
    /// Returns `None` if neither estimate is available.
    #[allow(dead_code)]
    #[must_use]
    pub fn eta_blended_s(&self, w_ewma: f64) -> Option<f64> {
        let w = w_ewma.clamp(0.0, 1.0);
        match (self.eta_ewma_s(), self.eta_windowed_s()) {
            (Some(a), Some(b)) => Some(w * a + (1.0 - w) * b),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    }

    /// Returns the current completion percentage ∈ [0.0, 100.0].
    #[allow(dead_code)]
    #[must_use]
    pub fn pct(&self) -> f64 {
        (self.frames_done as f64 / self.frames_total as f64 * 100.0).min(100.0)
    }

    /// Returns the number of frames remaining.
    #[allow(dead_code)]
    #[must_use]
    pub fn frames_remaining(&self) -> u64 {
        self.frames_total.saturating_sub(self.frames_done)
    }

    /// Returns `true` when all frames have been processed.
    #[allow(dead_code)]
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.frames_done >= self.frames_total
    }
}

// ── BitrateEtaEstimator ───────────────────────────────────────────────────────

/// ETA estimator based on observed output bitrate and estimated remaining
/// output size.
///
/// Useful when frame count is unknown but the target bitrate and input duration
/// are known (e.g., streaming conversion).
///
/// Estimate:
/// ```text
/// estimated_total_bytes = target_bitrate_bps * duration_seconds / 8
/// eta_s = bytes_remaining / ewma_write_rate
/// ```
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct BitrateEtaEstimator {
    /// Estimated total output bytes.
    pub estimated_total_bytes: u64,
    /// Bytes written so far.
    pub bytes_written: u64,
    /// Elapsed wall-clock milliseconds.
    pub elapsed_ms: u64,
    /// Observed byte-write rate in bytes/ms (EWMA).
    ewma_bps_ms: f64,
    /// Smoothing factor.
    alpha: f64,
}

impl BitrateEtaEstimator {
    /// Construct a new estimator from an estimated total output size.
    #[allow(dead_code)]
    #[must_use]
    pub fn new(estimated_total_bytes: u64, alpha: f64) -> Self {
        Self {
            estimated_total_bytes: estimated_total_bytes.max(1),
            bytes_written: 0,
            elapsed_ms: 0,
            ewma_bps_ms: 0.0,
            alpha: alpha.clamp(f64::EPSILON, 1.0),
        }
    }

    /// Update with the latest cumulative counters.
    #[allow(dead_code)]
    pub fn update(&mut self, bytes_written: u64, elapsed_ms: u64) {
        if elapsed_ms == 0 {
            return;
        }
        self.bytes_written = bytes_written;
        self.elapsed_ms = elapsed_ms;
        let instant_rate = bytes_written as f64 / elapsed_ms as f64;
        if self.ewma_bps_ms <= 0.0 {
            self.ewma_bps_ms = instant_rate;
        } else {
            self.ewma_bps_ms = self.alpha * instant_rate + (1.0 - self.alpha) * self.ewma_bps_ms;
        }
    }

    /// Current output rate in bytes per second.
    #[allow(dead_code)]
    #[must_use]
    pub fn bytes_per_second(&self) -> f64 {
        self.ewma_bps_ms * 1_000.0
    }

    /// Estimated remaining bytes.
    #[allow(dead_code)]
    #[must_use]
    pub fn bytes_remaining(&self) -> u64 {
        self.estimated_total_bytes
            .saturating_sub(self.bytes_written)
    }

    /// ETA in seconds.  Returns `None` until at least one update.
    #[allow(dead_code)]
    #[must_use]
    pub fn eta_s(&self) -> Option<f64> {
        if self.ewma_bps_ms <= 0.0 {
            return None;
        }
        let remaining = self.bytes_remaining();
        if remaining == 0 {
            return Some(0.0);
        }
        Some(remaining as f64 / (self.ewma_bps_ms * 1_000.0))
    }

    /// Completion percentage ∈ [0.0, 100.0].
    #[allow(dead_code)]
    #[must_use]
    pub fn pct(&self) -> f64 {
        (self.bytes_written as f64 / self.estimated_total_bytes as f64 * 100.0).min(100.0)
    }
}

// ── SpeedAdaptiveEta ──────────────────────────────────────────────────────────

/// Strategy used to derive an ETA estimate in [`SpeedAdaptiveEta`].
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EtaStrategy {
    /// Simple linear extrapolation from elapsed time and proportion done.
    Linear,
    /// Exponentially-weighted moving average over recent encoding speed samples.
    Ewma,
    /// Weighted blend of linear (40%) and EWMA (60%).
    Blended,
}

/// Multi-strategy ETA estimator that adapts to observed encoding speed.
///
/// Combines three complementary ETA strategies:
///
/// - **Linear**: `eta = elapsed * (total / done - 1)` — reliable early on.
/// - **EWMA**: sliding exponential average — reacts quickly to speed changes.
/// - **Blended**: weighted combination of linear and EWMA.
///
/// A confidence value in `[0.0, 1.0]` reflects how many samples have been
/// collected relative to the ring-buffer capacity.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SpeedAdaptiveEta {
    /// Total frames expected.
    pub total_frames: u64,
    /// Cumulative frames encoded.
    pub frames_done: u64,
    /// Cumulative wall-clock milliseconds elapsed.
    pub elapsed_ms: u64,
    /// EWMA smoothing factor α ∈ (0, 1].
    alpha: f64,
    /// Current EWMA FPS estimate.
    ewma_fps: f64,
    /// Ring buffer of recent `(frames, elapsed_ms)` samples.
    samples: Vec<(u64, u64)>,
    /// Write position in the ring buffer.
    sample_idx: usize,
    /// Number of valid entries in the ring buffer.
    sample_count: usize,
    /// Active estimation strategy.
    strategy: EtaStrategy,
}

impl SpeedAdaptiveEta {
    /// Maximum speed samples retained.
    pub const MAX_SAMPLES: usize = 32;

    /// Create a new estimator.
    ///
    /// * `total_frames` – expected total frame count (must be ≥ 1).
    /// * `alpha`        – EWMA smoothing factor; clamped to (0, 1].
    #[allow(dead_code)]
    #[must_use]
    pub fn new(total_frames: u64, alpha: f64) -> Self {
        Self {
            total_frames: total_frames.max(1),
            frames_done: 0,
            elapsed_ms: 0,
            alpha: alpha.clamp(f64::EPSILON, 1.0),
            ewma_fps: 0.0,
            samples: vec![(0, 0); Self::MAX_SAMPLES],
            sample_idx: 0,
            sample_count: 0,
            strategy: EtaStrategy::Blended,
        }
    }

    /// Create a new estimator with a specific strategy.
    #[allow(dead_code)]
    #[must_use]
    pub fn with_strategy(total_frames: u64, alpha: f64, strategy: EtaStrategy) -> Self {
        let mut s = Self::new(total_frames, alpha);
        s.strategy = strategy;
        s
    }

    /// Record a completed encoding segment.
    ///
    /// * `frames`     – frames encoded in this segment (0 is ignored).
    /// * `elapsed_ms` – wall-clock time for this segment in milliseconds.
    #[allow(dead_code)]
    pub fn record_segment(&mut self, frames: u64, elapsed_ms: u64) {
        if frames == 0 || elapsed_ms == 0 {
            return;
        }
        self.frames_done = (self.frames_done + frames).min(self.total_frames);
        self.elapsed_ms += elapsed_ms;

        let instant_fps = frames as f64 / (elapsed_ms as f64 / 1_000.0);
        if self.ewma_fps <= 0.0 {
            self.ewma_fps = instant_fps;
        } else {
            self.ewma_fps = self.alpha * instant_fps + (1.0 - self.alpha) * self.ewma_fps;
        }

        self.samples[self.sample_idx] = (frames, elapsed_ms);
        self.sample_idx = (self.sample_idx + 1) % Self::MAX_SAMPLES;
        self.sample_count = (self.sample_count + 1).min(Self::MAX_SAMPLES);
    }

    /// Frames remaining until completion.
    #[allow(dead_code)]
    #[must_use]
    pub fn frames_remaining(&self) -> u64 {
        self.total_frames.saturating_sub(self.frames_done)
    }

    /// Completion percentage ∈ [0.0, 100.0].
    #[allow(dead_code)]
    #[must_use]
    pub fn pct(&self) -> f64 {
        (self.frames_done as f64 / self.total_frames as f64 * 100.0).min(100.0)
    }

    /// Returns `true` when all frames have been processed.
    #[allow(dead_code)]
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.frames_done >= self.total_frames
    }

    /// Confidence in the current ETA estimate ∈ [0.0, 1.0].
    ///
    /// Reaches 1.0 only once the ring buffer is fully populated.
    #[allow(dead_code)]
    #[must_use]
    pub fn confidence(&self) -> f64 {
        if self.sample_count == 0 {
            return 0.0;
        }
        (self.sample_count as f64 / Self::MAX_SAMPLES as f64).min(1.0)
    }

    /// Linear ETA in seconds: extrapolate from elapsed time and proportion done.
    ///
    /// Returns `None` if no frames have been processed yet.
    #[allow(dead_code)]
    #[must_use]
    pub fn eta_linear_s(&self) -> Option<f64> {
        if self.frames_done == 0 || self.elapsed_ms == 0 {
            return None;
        }
        if self.frames_done >= self.total_frames {
            return Some(0.0);
        }
        let elapsed_s = self.elapsed_ms as f64 / 1_000.0;
        let total_s = elapsed_s * (self.total_frames as f64 / self.frames_done as f64);
        Some((total_s - elapsed_s).max(0.0))
    }

    /// EWMA-based ETA in seconds.
    ///
    /// Returns `None` until the first segment has been recorded.
    #[allow(dead_code)]
    #[must_use]
    pub fn eta_ewma_s(&self) -> Option<f64> {
        if self.ewma_fps <= 0.0 {
            return None;
        }
        let remaining = self.frames_remaining();
        if remaining == 0 {
            return Some(0.0);
        }
        Some(remaining as f64 / self.ewma_fps)
    }

    /// Blended ETA: 40% linear + 60% EWMA.
    ///
    /// Falls back to whichever estimate is available if only one can be computed.
    #[allow(dead_code)]
    #[must_use]
    pub fn eta_blended_s(&self) -> Option<f64> {
        match (self.eta_linear_s(), self.eta_ewma_s()) {
            (Some(lin), Some(ewma)) => Some(0.4 * lin + 0.6 * ewma),
            (Some(lin), None) => Some(lin),
            (None, Some(ewma)) => Some(ewma),
            (None, None) => None,
        }
    }

    /// ETA according to the configured strategy.
    #[allow(dead_code)]
    #[must_use]
    pub fn eta_s(&self) -> Option<f64> {
        match self.strategy {
            EtaStrategy::Linear => self.eta_linear_s(),
            EtaStrategy::Ewma => self.eta_ewma_s(),
            EtaStrategy::Blended => self.eta_blended_s(),
        }
    }

    /// Current EWMA FPS estimate (0.0 if no samples yet).
    #[allow(dead_code)]
    #[must_use]
    pub fn ewma_fps(&self) -> f64 {
        self.ewma_fps
    }

    /// Windowed-average FPS across all retained samples.
    #[allow(dead_code)]
    #[must_use]
    pub fn windowed_fps(&self) -> f64 {
        if self.sample_count == 0 {
            return 0.0;
        }
        let (total_f, total_ms): (u64, u64) = self.samples[..self.sample_count]
            .iter()
            .fold((0, 0), |(af, am), (f, ms)| (af + f, am + ms));
        if total_ms == 0 {
            return 0.0;
        }
        total_f as f64 / (total_ms as f64 / 1_000.0)
    }

    /// Override the active strategy.
    #[allow(dead_code)]
    pub fn set_strategy(&mut self, strategy: EtaStrategy) {
        self.strategy = strategy;
    }

    /// Returns the current strategy.
    #[allow(dead_code)]
    #[must_use]
    pub fn strategy(&self) -> EtaStrategy {
        self.strategy
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

    #[test]
    fn multi_overall_eta_none_without_active_fps() {
        let mut m = MultiJobProgress::new();
        m.add_job(ConversionProgress::new(1, 100));
        m.add_job(ConversionProgress::new(2, 200));
        assert!(m.overall_eta_s().is_none());
    }

    #[test]
    fn multi_overall_eta_computed() {
        let mut m = MultiJobProgress::new();
        let mut a = ConversionProgress::new(1, 1_000);
        // 500 frames in 2000 ms = 250 fps; 500 remaining → eta = 2s
        a.update(500, 0, 2_000);
        m.add_job(a);
        let eta = m.overall_eta_s().expect("should have ETA");
        assert!((eta - 2.0).abs() < 1e-6);
    }

    // ── EwmaProgress ─────────────────────────────────────────────────────────

    #[test]
    fn ewma_no_samples_returns_none_eta() {
        let ep = EwmaProgress::with_default_alpha(1_000);
        assert!(ep.eta_ewma_s().is_none());
        assert!(ep.eta_windowed_s().is_none());
        assert!(ep.eta_blended_s(0.5).is_none());
    }

    #[test]
    fn ewma_first_sample_seeds_directly() {
        let mut ep = EwmaProgress::new(1_000, 0.3);
        // 100 frames in 1000 ms = 100 fps
        ep.record_sample(100, 1_000);
        assert!((ep.ewma_fps() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn ewma_blends_second_sample() {
        let mut ep = EwmaProgress::new(1_000, 0.5);
        ep.record_sample(100, 1_000); // 100 fps seed
        ep.record_sample(200, 1_000); // 200 fps instant → blended = 0.5*200 + 0.5*100 = 150
        assert!((ep.ewma_fps() - 150.0).abs() < 1e-6);
    }

    #[test]
    fn ewma_pct_and_frames_remaining() {
        let mut ep = EwmaProgress::new(1_000, 0.3);
        ep.record_sample(400, 2_000);
        assert!((ep.pct() - 40.0).abs() < 1e-9);
        assert_eq!(ep.frames_remaining(), 600);
    }

    #[test]
    fn ewma_eta_decreases_as_frames_progress() {
        let mut ep = EwmaProgress::new(1_000, 1.0); // α=1 → no smoothing
        ep.record_sample(250, 1_000); // 250 fps; remaining=750; eta=3s
        let eta1 = ep.eta_ewma_s().expect("eta1");
        ep.record_sample(250, 1_000); // same speed; remaining=500; eta=2s
        let eta2 = ep.eta_ewma_s().expect("eta2");
        assert!(eta1 > eta2, "ETA should decrease as progress advances");
    }

    #[test]
    fn ewma_complete_returns_zero_eta() {
        let mut ep = EwmaProgress::new(100, 0.3);
        ep.record_sample(100, 1_000);
        assert_eq!(ep.eta_ewma_s(), Some(0.0));
        assert!(ep.is_complete());
    }

    #[test]
    fn ewma_windowed_fps_matches_history() {
        let mut ep = EwmaProgress::new(10_000, 0.3);
        // 3 segments each 100 frames / 500 ms = 200 fps
        for _ in 0..3 {
            ep.record_sample(100, 500);
        }
        let wfps = ep.windowed_fps();
        assert!((wfps - 200.0).abs() < 1e-6);
    }

    #[test]
    fn ewma_blended_uses_weight() {
        let mut ep = EwmaProgress::new(1_000, 0.3);
        ep.record_sample(200, 1_000); // seeds ewma
                                      // With w_ewma=1.0, blended should equal ewma
        let blended = ep.eta_blended_s(1.0);
        let ewma = ep.eta_ewma_s();
        assert_eq!(blended, ewma);
    }

    #[test]
    fn ewma_alpha_clamped_to_epsilon() {
        let mut ep = EwmaProgress::new(100, 0.0);
        ep.record_sample(10, 100);
        assert!(ep.ewma_fps() > 0.0);
    }

    // ── BitrateEtaEstimator ───────────────────────────────────────────────────

    #[test]
    fn bitrate_eta_no_update_returns_none() {
        let est = BitrateEtaEstimator::new(1_000_000, 0.3);
        assert!(est.eta_s().is_none());
    }

    #[test]
    fn bitrate_eta_first_update_seeds_rate() {
        let mut est = BitrateEtaEstimator::new(2_000_000, 1.0);
        // 1 MB in 1000 ms = 1000 bytes/ms
        est.update(1_000_000, 1_000);
        assert!((est.bytes_per_second() - 1_000_000.0).abs() < 1.0);
    }

    #[test]
    fn bitrate_eta_computed_correctly() {
        let mut est = BitrateEtaEstimator::new(2_000_000, 1.0);
        // 1 MB in 1000 ms → rate = 1 MB/s; remaining = 1 MB → 1 second
        est.update(1_000_000, 1_000);
        let eta = est.eta_s().expect("eta");
        assert!((eta - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bitrate_eta_pct_progress() {
        let mut est = BitrateEtaEstimator::new(4_000_000, 0.3);
        est.update(1_000_000, 500);
        assert!((est.pct() - 25.0).abs() < 1e-9);
    }

    #[test]
    fn bitrate_eta_zero_when_done() {
        let mut est = BitrateEtaEstimator::new(1_000_000, 1.0);
        est.update(1_000_000, 1_000);
        assert_eq!(est.eta_s(), Some(0.0));
    }

    #[test]
    fn bitrate_estimator_bytes_remaining() {
        let mut est = BitrateEtaEstimator::new(5_000, 0.3);
        est.update(2_000, 100);
        assert_eq!(est.bytes_remaining(), 3_000);
    }

    // ── SpeedAdaptiveEta ──────────────────────────────────────────────────────

    #[test]
    fn adaptive_no_samples_returns_none() {
        let eta = SpeedAdaptiveEta::new(1_000, 0.3);
        assert!(eta.eta_s().is_none());
        assert!(eta.eta_linear_s().is_none());
        assert!(eta.eta_ewma_s().is_none());
        assert!(eta.eta_blended_s().is_none());
    }

    #[test]
    fn adaptive_first_segment_seeds_ewma() {
        let mut eta = SpeedAdaptiveEta::new(1_000, 1.0);
        // 100 frames in 1000 ms = 100 fps
        eta.record_segment(100, 1_000);
        assert!((eta.ewma_fps() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn adaptive_eta_linear_correct() {
        let mut eta = SpeedAdaptiveEta::new(1_000, 0.3);
        // 500 frames in 2000 ms; elapsed=2s, ratio=1000/500=2, remaining=2s
        eta.record_segment(500, 2_000);
        let lin = eta.eta_linear_s().expect("linear eta");
        assert!((lin - 2.0).abs() < 1e-6);
    }

    #[test]
    fn adaptive_eta_ewma_correct() {
        let mut eta = SpeedAdaptiveEta::new(1_000, 1.0);
        // 500 frames in 1000 ms = 500 fps; remaining=500; eta=1s
        eta.record_segment(500, 1_000);
        let e = eta.eta_ewma_s().expect("ewma eta");
        assert!((e - 1.0).abs() < 1e-6);
    }

    #[test]
    fn adaptive_eta_zero_when_complete() {
        let mut eta = SpeedAdaptiveEta::new(100, 0.3);
        eta.record_segment(100, 1_000);
        assert_eq!(eta.eta_linear_s(), Some(0.0));
        assert_eq!(eta.eta_ewma_s(), Some(0.0));
        assert!(eta.is_complete());
    }

    #[test]
    fn adaptive_pct_correct() {
        let mut eta = SpeedAdaptiveEta::new(1_000, 0.3);
        eta.record_segment(250, 1_000);
        assert!((eta.pct() - 25.0).abs() < 1e-9);
    }

    #[test]
    fn adaptive_frames_remaining() {
        let mut eta = SpeedAdaptiveEta::new(1_000, 0.3);
        eta.record_segment(300, 1_000);
        assert_eq!(eta.frames_remaining(), 700);
    }

    #[test]
    fn adaptive_confidence_grows_with_samples() {
        let mut eta = SpeedAdaptiveEta::new(100_000, 0.3);
        assert_eq!(eta.confidence(), 0.0);
        eta.record_segment(100, 100);
        assert!(eta.confidence() > 0.0);
        for _ in 0..SpeedAdaptiveEta::MAX_SAMPLES {
            eta.record_segment(100, 100);
        }
        assert!((eta.confidence() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn adaptive_strategy_linear_uses_linear_only() {
        let mut eta = SpeedAdaptiveEta::with_strategy(1_000, 0.3, EtaStrategy::Linear);
        eta.record_segment(500, 2_000);
        assert_eq!(eta.eta_s(), eta.eta_linear_s());
    }

    #[test]
    fn adaptive_strategy_ewma_uses_ewma_only() {
        let mut eta = SpeedAdaptiveEta::with_strategy(1_000, 0.3, EtaStrategy::Ewma);
        eta.record_segment(500, 1_000);
        assert_eq!(eta.eta_s(), eta.eta_ewma_s());
    }

    #[test]
    fn adaptive_strategy_blended_combines_estimates() {
        let mut eta = SpeedAdaptiveEta::with_strategy(1_000, 1.0, EtaStrategy::Blended);
        eta.record_segment(500, 1_000);
        let lin = eta.eta_linear_s().expect("lin");
        let ewma = eta.eta_ewma_s().expect("ewma");
        let blended = eta.eta_blended_s().expect("blended");
        let expected = 0.4 * lin + 0.6 * ewma;
        assert!((blended - expected).abs() < 1e-6);
    }

    #[test]
    fn adaptive_set_strategy_changes_output() {
        let mut eta = SpeedAdaptiveEta::new(1_000, 1.0);
        eta.record_segment(500, 1_000);
        eta.set_strategy(EtaStrategy::Linear);
        assert!(eta.eta_s().is_some());
        eta.set_strategy(EtaStrategy::Ewma);
        assert!(eta.eta_s().is_some());
    }

    #[test]
    fn adaptive_windowed_fps_single_segment() {
        let mut eta = SpeedAdaptiveEta::new(10_000, 0.3);
        // 200 frames in 1000 ms = 200 fps
        eta.record_segment(200, 1_000);
        assert!((eta.windowed_fps() - 200.0).abs() < 1e-6);
    }

    #[test]
    fn adaptive_eta_decreases_as_frames_advance() {
        let mut eta = SpeedAdaptiveEta::with_strategy(1_000, 1.0, EtaStrategy::Ewma);
        eta.record_segment(250, 1_000);
        let e1 = eta.eta_ewma_s().expect("eta1");
        eta.record_segment(250, 1_000);
        let e2 = eta.eta_ewma_s().expect("eta2");
        assert!(e1 > e2, "ETA must decrease as frames are completed");
    }

    #[test]
    fn adaptive_zero_segment_ignored() {
        let mut eta = SpeedAdaptiveEta::new(1_000, 0.3);
        eta.record_segment(0, 0);
        assert_eq!(eta.frames_done, 0);
        assert!(eta.eta_s().is_none());
    }
}
