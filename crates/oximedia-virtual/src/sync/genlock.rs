//! Genlock synchronization
//!
//! Provides frame-accurate synchronization using genlock signals
//! for multi-camera and LED wall sync. Includes latency measurement
//! and compensation for the tracking-to-render pipeline.

use super::{SyncStatus, SyncTimestamp};
use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Genlock configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenlockConfig {
    /// Target frame rate
    pub frame_rate: f64,
    /// Sync tolerance in microseconds
    pub tolerance_us: u64,
    /// Enable auto-recovery
    pub auto_recovery: bool,
}

impl Default for GenlockConfig {
    fn default() -> Self {
        Self {
            frame_rate: 60.0,
            tolerance_us: 100,
            auto_recovery: true,
        }
    }
}

/// A single latency sample recording the delay at a specific pipeline stage.
#[derive(Debug, Clone, Copy)]
pub struct LatencySample {
    /// Stage identifier
    pub stage: PipelineStage,
    /// Measured latency
    pub latency: Duration,
    /// Frame number when this sample was recorded
    pub frame_number: u64,
    /// Timestamp when this sample was taken
    pub timestamp_ns: u64,
}

/// Pipeline stages where latency is measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PipelineStage {
    /// Camera tracking system (marker detection, IMU fusion)
    Tracking,
    /// Frustum computation from tracking data
    FrustumCompute,
    /// LED wall content rendering
    Render,
    /// Final compositing pass
    Composite,
    /// LED panel output (display scan-out)
    Display,
    /// Full end-to-end pipeline
    EndToEnd,
}

impl PipelineStage {
    /// Human-readable label for the stage.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Tracking => "Tracking",
            Self::FrustumCompute => "Frustum Compute",
            Self::Render => "Render",
            Self::Composite => "Composite",
            Self::Display => "Display",
            Self::EndToEnd => "End-to-End",
        }
    }
}

/// Latency statistics for a pipeline stage.
#[derive(Debug, Clone, Copy)]
pub struct LatencyStats {
    /// Minimum observed latency
    pub min: Duration,
    /// Maximum observed latency
    pub max: Duration,
    /// Mean latency
    pub mean: Duration,
    /// Approximate p95 latency (95th percentile)
    pub p95: Duration,
    /// Standard deviation in microseconds
    pub std_dev_us: f64,
    /// Number of samples used for these statistics
    pub sample_count: usize,
    /// Latest measured latency
    pub latest: Duration,
}

impl Default for LatencyStats {
    fn default() -> Self {
        Self {
            min: Duration::ZERO,
            max: Duration::ZERO,
            mean: Duration::ZERO,
            p95: Duration::ZERO,
            std_dev_us: 0.0,
            sample_count: 0,
            latest: Duration::ZERO,
        }
    }
}

/// Configuration for latency compensation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyCompensationConfig {
    /// Maximum number of latency samples to keep per stage (ring buffer).
    pub max_samples: usize,
    /// Enable predictive compensation (extrapolate tracking data forward).
    pub predictive_compensation: bool,
    /// Target total pipeline latency; compensation aims to bring actual
    /// latency below this value. In microseconds.
    pub target_latency_us: u64,
    /// Exponential moving average smoothing factor for latency estimates.
    /// 0.0 = no update; 1.0 = no smoothing.
    pub ema_alpha: f64,
    /// Whether to log warnings when latency exceeds target.
    pub warn_on_exceed: bool,
}

impl Default for LatencyCompensationConfig {
    fn default() -> Self {
        Self {
            max_samples: 256,
            predictive_compensation: true,
            target_latency_us: 16_667, // ~1 frame at 60 fps
            ema_alpha: 0.15,
            warn_on_exceed: true,
        }
    }
}

/// Latency tracker and compensator for the virtual production pipeline.
///
/// Records per-stage latency measurements, computes running statistics,
/// and provides a compensation offset that upstream modules (e.g. camera
/// tracking) can use to extrapolate data forward in time.
pub struct LatencyCompensator {
    config: LatencyCompensationConfig,
    /// Per-stage sample ring buffers.
    stage_samples: [VecDeque<Duration>; 6],
    /// EMA estimate per stage (in microseconds).
    ema_estimates_us: [f64; 6],
    /// Total samples recorded per stage.
    total_samples: [u64; 6],
    /// Latest sample per stage.
    latest: [Duration; 6],
}

impl LatencyCompensator {
    /// Create a new latency compensator.
    #[must_use]
    pub fn new(config: LatencyCompensationConfig) -> Self {
        let cap = config.max_samples;
        Self {
            config,
            stage_samples: [
                VecDeque::with_capacity(cap),
                VecDeque::with_capacity(cap),
                VecDeque::with_capacity(cap),
                VecDeque::with_capacity(cap),
                VecDeque::with_capacity(cap),
                VecDeque::with_capacity(cap),
            ],
            ema_estimates_us: [0.0; 6],
            total_samples: [0; 6],
            latest: [Duration::ZERO; 6],
        }
    }

    /// Map a pipeline stage to its index.
    fn stage_index(stage: PipelineStage) -> usize {
        match stage {
            PipelineStage::Tracking => 0,
            PipelineStage::FrustumCompute => 1,
            PipelineStage::Render => 2,
            PipelineStage::Composite => 3,
            PipelineStage::Display => 4,
            PipelineStage::EndToEnd => 5,
        }
    }

    /// Record a latency measurement for a pipeline stage.
    pub fn record(&mut self, stage: PipelineStage, latency: Duration) {
        let idx = Self::stage_index(stage);
        let buf = &mut self.stage_samples[idx];

        if buf.len() >= self.config.max_samples {
            buf.pop_front();
        }
        buf.push_back(latency);

        // Update EMA
        let sample_us = latency.as_micros() as f64;
        let alpha = self.config.ema_alpha;
        if self.total_samples[idx] == 0 {
            self.ema_estimates_us[idx] = sample_us;
        } else {
            self.ema_estimates_us[idx] =
                alpha * sample_us + (1.0 - alpha) * self.ema_estimates_us[idx];
        }

        self.total_samples[idx] += 1;
        self.latest[idx] = latency;
    }

    /// Get the EMA-smoothed latency estimate for a stage.
    #[must_use]
    pub fn estimated_latency(&self, stage: PipelineStage) -> Duration {
        let idx = Self::stage_index(stage);
        Duration::from_micros(self.ema_estimates_us[idx] as u64)
    }

    /// Compute the total estimated pipeline latency (sum of individual stages,
    /// excluding the EndToEnd stage which is measured separately).
    #[must_use]
    pub fn estimated_total_latency(&self) -> Duration {
        let sum_us: f64 = self.ema_estimates_us[0..5].iter().sum();
        Duration::from_micros(sum_us as u64)
    }

    /// Compute compensation offset: how far ahead (in time) to extrapolate
    /// tracking data to compensate for the pipeline latency.
    ///
    /// Returns `Duration::ZERO` if predictive compensation is disabled or
    /// if no samples have been recorded.
    #[must_use]
    pub fn compensation_offset(&self) -> Duration {
        if !self.config.predictive_compensation {
            return Duration::ZERO;
        }

        let total_us: f64 = self.ema_estimates_us[0..5].iter().sum();
        if total_us <= 0.0 {
            return Duration::ZERO;
        }

        // Compensation is capped at 2x the target latency to avoid
        // over-extrapolation in pathological cases.
        let cap_us = (self.config.target_latency_us as f64) * 2.0;
        let offset_us = total_us.min(cap_us);
        Duration::from_micros(offset_us as u64)
    }

    /// Check whether the current estimated pipeline latency exceeds the target.
    #[must_use]
    pub fn is_over_budget(&self) -> bool {
        let total_us: f64 = self.ema_estimates_us[0..5].iter().sum();
        total_us > self.config.target_latency_us as f64
    }

    /// Compute full statistics for a pipeline stage.
    #[must_use]
    pub fn stats(&self, stage: PipelineStage) -> LatencyStats {
        let idx = Self::stage_index(stage);
        let buf = &self.stage_samples[idx];

        if buf.is_empty() {
            return LatencyStats::default();
        }

        let count = buf.len();
        let mut sum_us: f64 = 0.0;
        let mut min = Duration::MAX;
        let mut max = Duration::ZERO;

        for &d in buf {
            sum_us += d.as_micros() as f64;
            if d < min {
                min = d;
            }
            if d > max {
                max = d;
            }
        }

        let mean_us = sum_us / count as f64;
        let mean = Duration::from_micros(mean_us as u64);

        // Standard deviation
        let variance: f64 = buf
            .iter()
            .map(|d| {
                let diff = d.as_micros() as f64 - mean_us;
                diff * diff
            })
            .sum::<f64>()
            / count as f64;
        let std_dev_us = variance.sqrt();

        // P95: sort a copy and pick the 95th percentile element
        let p95 = {
            let mut sorted: Vec<Duration> = buf.iter().copied().collect();
            sorted.sort();
            let p95_idx = ((count as f64 * 0.95) as usize).min(count.saturating_sub(1));
            sorted[p95_idx]
        };

        LatencyStats {
            min,
            max,
            mean,
            p95,
            std_dev_us,
            sample_count: count,
            latest: self.latest[idx],
        }
    }

    /// Return the number of samples recorded for a stage.
    #[must_use]
    pub fn sample_count(&self, stage: PipelineStage) -> u64 {
        self.total_samples[Self::stage_index(stage)]
    }

    /// Reset all recorded samples and estimates.
    pub fn reset(&mut self) {
        for buf in &mut self.stage_samples {
            buf.clear();
        }
        self.ema_estimates_us = [0.0; 6];
        self.total_samples = [0; 6];
        self.latest = [Duration::ZERO; 6];
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &LatencyCompensationConfig {
        &self.config
    }
}

/// Genlock synchronization
pub struct GenlockSync {
    config: GenlockConfig,
    status: SyncStatus,
    reference_time: Option<Instant>,
    frame_count: u64,
    last_sync: Option<SyncTimestamp>,
    /// Latency compensator for the tracking-to-render pipeline
    latency_compensator: LatencyCompensator,
}

impl GenlockSync {
    /// Create new genlock sync
    pub fn new(config: GenlockConfig) -> Result<Self> {
        Ok(Self {
            config,
            status: SyncStatus::Unlocked,
            reference_time: None,
            frame_count: 0,
            last_sync: None,
            latency_compensator: LatencyCompensator::new(LatencyCompensationConfig::default()),
        })
    }

    /// Create genlock sync with a custom latency compensation configuration.
    pub fn with_latency_config(
        config: GenlockConfig,
        latency_config: LatencyCompensationConfig,
    ) -> Result<Self> {
        Ok(Self {
            config,
            status: SyncStatus::Unlocked,
            reference_time: None,
            frame_count: 0,
            last_sync: None,
            latency_compensator: LatencyCompensator::new(latency_config),
        })
    }

    /// Wait for next frame sync
    pub fn wait_for_frame(&mut self) -> Result<SyncTimestamp> {
        let now = Instant::now();

        // Initialize reference time on first call
        if self.reference_time.is_none() {
            self.reference_time = Some(now);
            self.status = SyncStatus::Locking;
        }

        let reference = *self.reference_time.get_or_insert(now);
        let frame_duration = Duration::from_secs_f64(1.0 / self.config.frame_rate);

        // Calculate target time for this frame
        let target_time = reference + frame_duration * self.frame_count as u32;

        // Wait if we're early
        if now < target_time {
            let wait_time = target_time.duration_since(now);
            std::thread::sleep(wait_time);
        }

        // Check sync status
        let actual_time = Instant::now();
        let offset = if actual_time >= target_time {
            actual_time.duration_since(target_time)
        } else {
            Duration::ZERO
        };

        if offset.as_micros() as u64 > self.config.tolerance_us {
            self.status = SyncStatus::Locking;
            if self.config.auto_recovery {
                // Reset reference time to recover
                self.reference_time = Some(actual_time);
                self.frame_count = 0;
            }
        } else {
            self.status = SyncStatus::Locked;
        }

        let timestamp = SyncTimestamp::new(
            actual_time.duration_since(reference).as_nanos() as u64,
            self.frame_count,
        );

        self.last_sync = Some(timestamp);
        self.frame_count += 1;

        Ok(timestamp)
    }

    /// Record a latency measurement for a pipeline stage.
    pub fn record_latency(&mut self, stage: PipelineStage, latency: Duration) {
        self.latency_compensator.record(stage, latency);
    }

    /// Get the latency compensation offset for upstream modules.
    #[must_use]
    pub fn compensation_offset(&self) -> Duration {
        self.latency_compensator.compensation_offset()
    }

    /// Get latency statistics for a pipeline stage.
    #[must_use]
    pub fn latency_stats(&self, stage: PipelineStage) -> LatencyStats {
        self.latency_compensator.stats(stage)
    }

    /// Check whether the pipeline is over its latency budget.
    #[must_use]
    pub fn is_latency_over_budget(&self) -> bool {
        self.latency_compensator.is_over_budget()
    }

    /// Get the latency compensator for direct access.
    #[must_use]
    pub fn latency_compensator(&self) -> &LatencyCompensator {
        &self.latency_compensator
    }

    /// Get mutable access to the latency compensator.
    pub fn latency_compensator_mut(&mut self) -> &mut LatencyCompensator {
        &mut self.latency_compensator
    }

    /// Get sync status
    #[must_use]
    pub fn status(&self) -> SyncStatus {
        self.status
    }

    /// Get current frame count
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Reset synchronization
    pub fn reset(&mut self) {
        self.reference_time = None;
        self.frame_count = 0;
        self.status = SyncStatus::Unlocked;
        self.last_sync = None;
        self.latency_compensator.reset();
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &GenlockConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genlock_creation() {
        let config = GenlockConfig::default();
        let genlock = GenlockSync::new(config);
        assert!(genlock.is_ok());
    }

    #[test]
    fn test_genlock_status() {
        let config = GenlockConfig::default();
        let genlock = GenlockSync::new(config).expect("should succeed in test");
        assert_eq!(genlock.status(), SyncStatus::Unlocked);
    }

    #[test]
    fn test_genlock_reset() {
        let config = GenlockConfig::default();
        let mut genlock = GenlockSync::new(config).expect("should succeed in test");
        genlock.reset();
        assert_eq!(genlock.frame_count(), 0);
        assert_eq!(genlock.status(), SyncStatus::Unlocked);
    }

    // --- Latency compensator tests ---

    #[test]
    fn test_latency_compensator_creation() {
        let comp = LatencyCompensator::new(LatencyCompensationConfig::default());
        assert_eq!(comp.sample_count(PipelineStage::Tracking), 0);
        assert_eq!(
            comp.estimated_latency(PipelineStage::Tracking),
            Duration::ZERO
        );
    }

    #[test]
    fn test_latency_record_single_sample() {
        let mut comp = LatencyCompensator::new(LatencyCompensationConfig::default());
        let latency = Duration::from_millis(5);
        comp.record(PipelineStage::Tracking, latency);

        assert_eq!(comp.sample_count(PipelineStage::Tracking), 1);
        // First sample sets EMA directly
        assert_eq!(comp.estimated_latency(PipelineStage::Tracking), latency);
    }

    #[test]
    fn test_latency_ema_smoothing() {
        let config = LatencyCompensationConfig {
            ema_alpha: 0.5,
            ..LatencyCompensationConfig::default()
        };
        let mut comp = LatencyCompensator::new(config);

        comp.record(PipelineStage::Render, Duration::from_millis(1));
        comp.record(PipelineStage::Render, Duration::from_millis(2));

        // EMA: 0.5 * 2000 + 0.5 * 1000 = 1500
        let est = comp.estimated_latency(PipelineStage::Render);
        assert_eq!(est, Duration::from_micros(1500));
    }

    #[test]
    fn test_latency_stats_computation() {
        let mut comp = LatencyCompensator::new(LatencyCompensationConfig::default());

        // Record 10 samples with known values
        for i in 1..=10 {
            comp.record(PipelineStage::Composite, Duration::from_micros(i * 100));
        }

        let stats = comp.stats(PipelineStage::Composite);
        assert_eq!(stats.sample_count, 10);
        assert_eq!(stats.min, Duration::from_micros(100));
        assert_eq!(stats.max, Duration::from_millis(1));
        // Mean should be 550
        assert_eq!(stats.mean, Duration::from_micros(550));
        // Latest should be 1000
        assert_eq!(stats.latest, Duration::from_millis(1));
        // P95 should be the 10th element (index 9) => 1000us
        assert!(stats.p95 >= Duration::from_micros(900));
    }

    #[test]
    fn test_latency_stats_std_dev() {
        let mut comp = LatencyCompensator::new(LatencyCompensationConfig::default());

        // All same value => std_dev should be ~0
        for _ in 0..20 {
            comp.record(PipelineStage::Display, Duration::from_micros(500));
        }
        let stats = comp.stats(PipelineStage::Display);
        assert!(
            stats.std_dev_us < 0.01,
            "std_dev should be ~0: {}",
            stats.std_dev_us
        );
    }

    #[test]
    fn test_latency_stats_empty() {
        let comp = LatencyCompensator::new(LatencyCompensationConfig::default());
        let stats = comp.stats(PipelineStage::Tracking);
        assert_eq!(stats.sample_count, 0);
        assert_eq!(stats.min, Duration::ZERO);
        assert_eq!(stats.max, Duration::ZERO);
    }

    #[test]
    fn test_latency_ring_buffer_overflow() {
        let config = LatencyCompensationConfig {
            max_samples: 4,
            ..LatencyCompensationConfig::default()
        };
        let mut comp = LatencyCompensator::new(config);

        for i in 0..10 {
            comp.record(PipelineStage::Tracking, Duration::from_micros(i * 100));
        }

        // Buffer should only contain the last 4 samples
        let stats = comp.stats(PipelineStage::Tracking);
        assert_eq!(stats.sample_count, 4);
        assert_eq!(stats.min, Duration::from_micros(600));
        assert_eq!(stats.max, Duration::from_micros(900));
    }

    #[test]
    fn test_estimated_total_latency() {
        let mut comp = LatencyCompensator::new(LatencyCompensationConfig::default());

        comp.record(PipelineStage::Tracking, Duration::from_millis(2));
        comp.record(PipelineStage::FrustumCompute, Duration::from_millis(1));
        comp.record(PipelineStage::Render, Duration::from_millis(5));
        comp.record(PipelineStage::Composite, Duration::from_millis(3));
        comp.record(PipelineStage::Display, Duration::from_millis(1));

        let total = comp.estimated_total_latency();
        assert_eq!(total, Duration::from_millis(12));
    }

    #[test]
    fn test_compensation_offset_basic() {
        let config = LatencyCompensationConfig {
            predictive_compensation: true,
            target_latency_us: 16_667,
            ..LatencyCompensationConfig::default()
        };
        let mut comp = LatencyCompensator::new(config);

        comp.record(PipelineStage::Tracking, Duration::from_millis(5));
        comp.record(PipelineStage::Render, Duration::from_millis(8));

        let offset = comp.compensation_offset();
        // Should be sum of stages: 5000 + 8000 = 13000us
        assert_eq!(offset, Duration::from_millis(13));
    }

    #[test]
    fn test_compensation_offset_capped() {
        let config = LatencyCompensationConfig {
            predictive_compensation: true,
            target_latency_us: 5000,
            ..LatencyCompensationConfig::default()
        };
        let mut comp = LatencyCompensator::new(config);

        // Record latencies that sum to much more than 2x target
        comp.record(PipelineStage::Tracking, Duration::from_millis(10));
        comp.record(PipelineStage::Render, Duration::from_millis(10));

        let offset = comp.compensation_offset();
        // Should be capped at 2 * 5000 = 10000us
        assert_eq!(offset, Duration::from_millis(10));
    }

    #[test]
    fn test_compensation_offset_disabled() {
        let config = LatencyCompensationConfig {
            predictive_compensation: false,
            ..LatencyCompensationConfig::default()
        };
        let mut comp = LatencyCompensator::new(config);

        comp.record(PipelineStage::Tracking, Duration::from_millis(5));

        let offset = comp.compensation_offset();
        assert_eq!(offset, Duration::ZERO);
    }

    #[test]
    fn test_is_over_budget() {
        let config = LatencyCompensationConfig {
            target_latency_us: 10_000,
            ..LatencyCompensationConfig::default()
        };
        let mut comp = LatencyCompensator::new(config);

        comp.record(PipelineStage::Tracking, Duration::from_millis(3));
        assert!(!comp.is_over_budget());

        comp.record(PipelineStage::Render, Duration::from_millis(8));
        assert!(comp.is_over_budget());
    }

    #[test]
    fn test_latency_compensator_reset() {
        let mut comp = LatencyCompensator::new(LatencyCompensationConfig::default());
        comp.record(PipelineStage::Tracking, Duration::from_millis(5));
        comp.record(PipelineStage::Render, Duration::from_millis(5));

        comp.reset();

        assert_eq!(comp.sample_count(PipelineStage::Tracking), 0);
        assert_eq!(comp.sample_count(PipelineStage::Render), 0);
        assert_eq!(
            comp.estimated_latency(PipelineStage::Tracking),
            Duration::ZERO
        );
    }

    #[test]
    fn test_genlock_with_latency_config() {
        let genlock_config = GenlockConfig::default();
        let latency_config = LatencyCompensationConfig {
            target_latency_us: 8000,
            ..LatencyCompensationConfig::default()
        };

        let mut genlock = GenlockSync::with_latency_config(genlock_config, latency_config)
            .expect("should succeed in test");

        genlock.record_latency(PipelineStage::Tracking, Duration::from_millis(3));
        genlock.record_latency(PipelineStage::Render, Duration::from_millis(6));

        assert!(genlock.is_latency_over_budget());
        assert!(genlock.compensation_offset() > Duration::ZERO);

        let stats = genlock.latency_stats(PipelineStage::Tracking);
        assert_eq!(stats.sample_count, 1);
    }

    #[test]
    fn test_genlock_reset_clears_latency() {
        let mut genlock =
            GenlockSync::new(GenlockConfig::default()).expect("should succeed in test");
        genlock.record_latency(PipelineStage::Render, Duration::from_millis(5));

        genlock.reset();

        let stats = genlock.latency_stats(PipelineStage::Render);
        assert_eq!(stats.sample_count, 0);
    }

    #[test]
    fn test_pipeline_stage_labels() {
        assert_eq!(PipelineStage::Tracking.label(), "Tracking");
        assert_eq!(PipelineStage::FrustumCompute.label(), "Frustum Compute");
        assert_eq!(PipelineStage::Render.label(), "Render");
        assert_eq!(PipelineStage::Composite.label(), "Composite");
        assert_eq!(PipelineStage::Display.label(), "Display");
        assert_eq!(PipelineStage::EndToEnd.label(), "End-to-End");
    }

    #[test]
    fn test_latency_ema_converges() {
        let config = LatencyCompensationConfig {
            ema_alpha: 0.2,
            ..LatencyCompensationConfig::default()
        };
        let mut comp = LatencyCompensator::new(config);

        // Feed constant value; EMA should converge to it
        for _ in 0..100 {
            comp.record(PipelineStage::Tracking, Duration::from_millis(1));
        }

        let est = comp.estimated_latency(PipelineStage::Tracking);
        let diff = if est > Duration::from_millis(1) {
            est - Duration::from_millis(1)
        } else {
            Duration::from_millis(1) - est
        };
        assert!(
            diff < Duration::from_micros(5),
            "EMA should converge: {est:?}"
        );
    }

    #[test]
    fn test_per_stage_independence() {
        let mut comp = LatencyCompensator::new(LatencyCompensationConfig::default());

        comp.record(PipelineStage::Tracking, Duration::from_millis(1));
        comp.record(PipelineStage::Render, Duration::from_millis(9));

        assert_eq!(
            comp.estimated_latency(PipelineStage::Tracking),
            Duration::from_millis(1)
        );
        assert_eq!(
            comp.estimated_latency(PipelineStage::Render),
            Duration::from_millis(9)
        );
        // Other stages should still be zero
        assert_eq!(
            comp.estimated_latency(PipelineStage::Composite),
            Duration::ZERO
        );
    }
}
