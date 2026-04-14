// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Render progress tracking with ETA prediction per worker class.
//!
//! This module provides:
//!
//! - [`WorkerClass`] — classification of worker hardware tiers (CPU / GPU / HighEnd GPU)
//! - [`CompletionSample`] — a single frame/chunk completion record for a specific worker class
//! - [`EtaPredictor`] — accumulates historical completion rates per worker class and
//!   uses exponential moving averages (EMA) to forecast completion time
//! - [`ProgressReport`] — snapshot of a job's current progress with ETA in seconds,
//!   confidence interval, and per-worker-class throughput breakdown
//!
//! ## Design approach
//!
//! Naive ETA estimators simply divide remaining work by the observed average frame
//! time.  This module goes further:
//!
//! 1. **Per-worker-class EMA** — heterogeneous farms have GPU-accelerated nodes that
//!    are 5–20× faster than CPU nodes.  A single global average produces wildly
//!    inaccurate ETAs when the mix of workers changes.  By tracking completion rates
//!    separately for each [`WorkerClass`], the predictor can weight each class by the
//!    number of workers of that class currently assigned to the job.
//!
//! 2. **Exponential moving average** — recent frame times matter more than old ones
//!    (e.g. after a scene complexity change).  The EMA weight `alpha` controls how
//!    quickly the estimate adapts (default 0.3).
//!
//! 3. **Variance / confidence interval** — the predictor also tracks the EMA variance
//!    of frame times so it can report a 95 % confidence interval around the ETA.
//!
//! 4. **Stallout detection** — if the observed rate for a class drops to near-zero the
//!    predictor surfaces this as a potential stall rather than returning an implausibly
//!    large ETA.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// WorkerClass
// ─────────────────────────────────────────────────────────────────────────────

/// Hardware tier classification for render workers.
///
/// Grouping workers into classes prevents per-worker noise from distorting the
/// ETA estimate while still capturing the major performance tiers found in
/// production render farms.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkerClass {
    /// CPU-only render node (no GPU acceleration).
    CpuOnly,
    /// Entry-level GPU worker (e.g. NVIDIA RTX 3060 / A2000).
    GpuStandard,
    /// High-end GPU worker (e.g. NVIDIA A100 / H100, RTX 4090).
    GpuHighEnd,
    /// Cloud preemptible / spot instance (variable performance).
    CloudSpot,
    /// Custom class identified by a user-defined string label.
    Custom(String),
}

impl WorkerClass {
    /// A human-readable display name.
    #[must_use]
    pub fn display_name(&self) -> &str {
        match self {
            Self::CpuOnly => "CPU-Only",
            Self::GpuStandard => "GPU-Standard",
            Self::GpuHighEnd => "GPU-HighEnd",
            Self::CloudSpot => "Cloud-Spot",
            Self::Custom(s) => s.as_str(),
        }
    }

    /// Relative performance weight used as a prior when no samples are available
    /// (CPU = 1.0 baseline).
    #[must_use]
    pub fn baseline_weight(&self) -> f64 {
        match self {
            Self::CpuOnly => 1.0,
            Self::GpuStandard => 5.0,
            Self::GpuHighEnd => 15.0,
            Self::CloudSpot => 0.8,
            Self::Custom(_) => 1.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CompletionSample
// ─────────────────────────────────────────────────────────────────────────────

/// A single frame (or chunk) completion record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionSample {
    /// Which worker class produced this sample.
    pub worker_class: WorkerClass,
    /// Wall-clock seconds taken to render this frame / chunk.
    pub duration_secs: f64,
    /// Frame number (for ordering / diagnostic purposes).
    pub frame: u32,
    /// Worker instance identifier.
    pub worker_id: String,
}

impl CompletionSample {
    /// Construct a new completion sample.
    ///
    /// `duration_secs` is clamped to a minimum of 0.001 s to avoid
    /// divide-by-zero issues in throughput calculations.
    #[must_use]
    pub fn new(
        worker_class: WorkerClass,
        duration_secs: f64,
        frame: u32,
        worker_id: impl Into<String>,
    ) -> Self {
        Self {
            worker_class,
            duration_secs: duration_secs.max(1e-3),
            frame,
            worker_id: worker_id.into(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ClassStats — internal per-class EMA accumulator
// ─────────────────────────────────────────────────────────────────────────────

/// Internal EMA state for one worker class.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClassStats {
    /// EMA of seconds-per-frame.
    ema_secs_per_frame: f64,
    /// EMA of the squared deviation from the mean (Welford-style EMA variance).
    ema_variance: f64,
    /// Total samples ingested.
    sample_count: u64,
    /// EMA smoothing factor (0 < alpha ≤ 1).
    alpha: f64,
}

impl ClassStats {
    fn new(alpha: f64) -> Self {
        Self {
            ema_secs_per_frame: 0.0,
            ema_variance: 0.0,
            sample_count: 0,
            alpha: alpha.clamp(1e-6, 1.0),
        }
    }

    /// Ingest a new duration sample, updating EMA and variance.
    fn update(&mut self, duration_secs: f64) {
        if self.sample_count == 0 {
            // Cold start: seed directly with first sample
            self.ema_secs_per_frame = duration_secs;
            self.ema_variance = 0.0;
        } else {
            let diff = duration_secs - self.ema_secs_per_frame;
            self.ema_secs_per_frame += self.alpha * diff;
            // EMA variance (Welford-style): V_new = (1-α)*(V_old + α*diff²)
            self.ema_variance = (1.0 - self.alpha) * (self.ema_variance + self.alpha * diff * diff);
        }
        self.sample_count += 1;
    }

    /// EMA standard deviation in seconds.
    fn std_dev(&self) -> f64 {
        self.ema_variance.max(0.0).sqrt()
    }

    /// Frames per second throughput.
    fn frames_per_second(&self) -> f64 {
        if self.ema_secs_per_frame <= 0.0 {
            0.0
        } else {
            1.0 / self.ema_secs_per_frame
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WorkerClassAllocation
// ─────────────────────────────────────────────────────────────────────────────

/// How many workers of each class are currently assigned to a job.
///
/// Passed to [`EtaPredictor::predict`] so the ETA can weight each class
/// proportionally to its current worker count.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkerClassAllocation {
    counts: HashMap<WorkerClass, u32>,
}

impl WorkerClassAllocation {
    /// Create an empty allocation.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of workers for a class.
    pub fn set(&mut self, class: WorkerClass, count: u32) {
        if count == 0 {
            self.counts.remove(&class);
        } else {
            self.counts.insert(class, count);
        }
    }

    /// Increment the count for a class by 1.
    pub fn add_one(&mut self, class: WorkerClass) {
        *self.counts.entry(class).or_insert(0) += 1;
    }

    /// Total workers across all classes.
    #[must_use]
    pub fn total_workers(&self) -> u32 {
        self.counts.values().sum()
    }

    /// Iterator over (class, count) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&WorkerClass, u32)> {
        self.counts.iter().map(|(k, &v)| (k, v))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EtaPredictor
// ─────────────────────────────────────────────────────────────────────────────

/// Predicts completion ETA for a render job using per-worker-class historical rates.
///
/// ## Usage
///
/// ```rust,ignore
/// let mut predictor = EtaPredictor::new(1000); // 1000 total frames
///
/// // Feed completion samples as frames finish
/// predictor.ingest(CompletionSample::new(WorkerClass::GpuHighEnd, 0.5, 1, "gpu-01"));
/// predictor.ingest(CompletionSample::new(WorkerClass::CpuOnly, 8.0, 2, "cpu-01"));
///
/// // Describe the current worker mix
/// let mut alloc = WorkerClassAllocation::new();
/// alloc.set(WorkerClass::GpuHighEnd, 2);
/// alloc.set(WorkerClass::CpuOnly, 4);
///
/// let report = predictor.predict(&alloc);
/// println!("ETA: {} seconds", report.eta_secs);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EtaPredictor {
    /// Total frames in the job.
    total_frames: u32,
    /// Number of frames completed so far.
    completed_frames: u32,
    /// Number of frames that failed (not retried yet).
    failed_frames: u32,
    /// Per-class EMA stats.
    class_stats: HashMap<WorkerClass, ClassStats>,
    /// EMA smoothing factor applied to new classes.
    alpha: f64,
    /// Stall threshold: if the weighted throughput is below this (frames/sec),
    /// the prediction is flagged as potentially stalled.
    stall_threshold_fps: f64,
}

impl EtaPredictor {
    /// Create a predictor with `total_frames` and default alpha of 0.3.
    #[must_use]
    pub fn new(total_frames: u32) -> Self {
        Self::with_alpha(total_frames, 0.3)
    }

    /// Create a predictor with a custom EMA alpha.
    ///
    /// `alpha` is clamped to (0.0, 1.0].
    #[must_use]
    pub fn with_alpha(total_frames: u32, alpha: f64) -> Self {
        Self {
            total_frames,
            completed_frames: 0,
            failed_frames: 0,
            class_stats: HashMap::new(),
            alpha: alpha.clamp(1e-6, 1.0),
            stall_threshold_fps: 1e-4,
        }
    }

    /// Ingest a new completion sample, updating the EMA for its worker class.
    pub fn ingest(&mut self, sample: CompletionSample) {
        self.completed_frames += 1;
        let alpha = self.alpha;
        let stats = self
            .class_stats
            .entry(sample.worker_class)
            .or_insert_with(|| ClassStats::new(alpha));
        stats.update(sample.duration_secs);
    }

    /// Record a frame failure (excluded from throughput calculation).
    pub fn record_failure(&mut self) {
        self.failed_frames += 1;
    }

    /// Progress fraction in [0.0, 1.0].
    #[must_use]
    pub fn progress(&self) -> f64 {
        if self.total_frames == 0 {
            return 1.0;
        }
        f64::from(self.completed_frames) / f64::from(self.total_frames)
    }

    /// Number of frames remaining.
    #[must_use]
    pub fn remaining_frames(&self) -> u32 {
        self.total_frames.saturating_sub(self.completed_frames)
    }

    /// Effective aggregate throughput in frames/second given a worker allocation.
    ///
    /// Throughput is the sum over classes of:
    ///   `class_fps × worker_count_for_class`
    ///
    /// Classes with no observed samples fall back to the baseline weight ratio
    /// relative to a single CPU worker (if CPU stats are available).
    #[must_use]
    pub fn aggregate_fps(&self, allocation: &WorkerClassAllocation) -> f64 {
        let cpu_fps = self
            .class_stats
            .get(&WorkerClass::CpuOnly)
            .map(|s| s.frames_per_second());

        let mut total_fps = 0.0_f64;

        for (class, count) in allocation.iter() {
            let fps = if let Some(stats) = self.class_stats.get(class) {
                if stats.sample_count > 0 {
                    stats.frames_per_second()
                } else {
                    // Has a stats entry but no samples yet — use baseline weight
                    self.fps_from_baseline(class, cpu_fps)
                }
            } else {
                // No stats entry at all — use baseline weight
                self.fps_from_baseline(class, cpu_fps)
            };

            total_fps += fps * f64::from(count);
        }

        total_fps
    }

    /// Derive a best-guess fps for a class with no samples, using its baseline weight.
    fn fps_from_baseline(&self, class: &WorkerClass, cpu_fps: Option<f64>) -> f64 {
        match cpu_fps {
            Some(base) => base * class.baseline_weight(),
            None => {
                // No CPU baseline either: use a conservative default of 1 frame/30 s
                (1.0 / 30.0) * class.baseline_weight()
            }
        }
    }

    /// Weighted EMA standard deviation across all classes weighted by worker count.
    fn aggregate_std_dev(&self, allocation: &WorkerClassAllocation) -> f64 {
        let total = allocation.total_workers();
        if total == 0 {
            return 0.0;
        }
        let mut weighted_var = 0.0_f64;
        for (class, count) in allocation.iter() {
            let std = self
                .class_stats
                .get(class)
                .map(|s| s.std_dev())
                .unwrap_or(0.0);
            let weight = f64::from(count) / f64::from(total);
            weighted_var += weight * std * std;
        }
        weighted_var.max(0.0).sqrt()
    }

    /// Predict ETA and build a [`ProgressReport`].
    ///
    /// If the aggregate throughput is zero (no workers allocated or no samples yet),
    /// `eta_secs` is `f64::INFINITY` and `is_stalled` is `true`.
    #[must_use]
    pub fn predict(&self, allocation: &WorkerClassAllocation) -> ProgressReport {
        let remaining = self.remaining_frames();
        let fps = self.aggregate_fps(allocation);

        let (eta_secs, is_stalled, confidence_interval_secs) =
            if fps <= self.stall_threshold_fps || remaining == 0 {
                let stalled = fps <= self.stall_threshold_fps && remaining > 0;
                (
                    if remaining == 0 { 0.0 } else { f64::INFINITY },
                    stalled,
                    0.0,
                )
            } else {
                let eta = f64::from(remaining) / fps;
                // 95% CI: ETA ± 1.96 * std_dev_per_frame * sqrt(remaining) / fps
                // Simplified: CI ≈ 1.96 * (std_dev / fps) * sqrt(remaining)
                let std = self.aggregate_std_dev(allocation);
                let ci = 1.96 * (std / fps) * (f64::from(remaining).sqrt());
                (eta, false, ci)
            };

        // Per-class throughput snapshot
        let class_throughput: HashMap<String, f64> = self
            .class_stats
            .iter()
            .map(|(class, stats)| (class.display_name().to_owned(), stats.frames_per_second()))
            .collect();

        ProgressReport {
            total_frames: self.total_frames,
            completed_frames: self.completed_frames,
            failed_frames: self.failed_frames,
            remaining_frames: remaining,
            progress_fraction: self.progress(),
            aggregate_fps: fps,
            eta_secs,
            confidence_interval_secs,
            is_stalled,
            class_throughput_fps: class_throughput,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ProgressReport
// ─────────────────────────────────────────────────────────────────────────────

/// Snapshot of a job's current progress with ETA prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressReport {
    /// Total frames in the job.
    pub total_frames: u32,
    /// Frames completed so far.
    pub completed_frames: u32,
    /// Frames that failed.
    pub failed_frames: u32,
    /// Frames remaining to render.
    pub remaining_frames: u32,
    /// Progress fraction in [0.0, 1.0].
    pub progress_fraction: f64,
    /// Aggregate frames-per-second throughput across all allocated workers.
    pub aggregate_fps: f64,
    /// Predicted seconds until job completion.  `f64::INFINITY` if the job is stalled.
    pub eta_secs: f64,
    /// Half-width of the 95% confidence interval around `eta_secs`, in seconds.
    ///
    /// The true ETA is expected to lie in `[eta_secs - ci, eta_secs + ci]` with
    /// ~95% probability, assuming normally distributed frame times.
    pub confidence_interval_secs: f64,
    /// Whether the predictor suspects the job has stalled (throughput near zero).
    pub is_stalled: bool,
    /// Per-worker-class throughput in frames/second at the time of this report.
    pub class_throughput_fps: HashMap<String, f64>,
}

impl ProgressReport {
    /// Whether the job is complete (remaining_frames == 0).
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.remaining_frames == 0
    }

    /// ETA formatted as `HH:MM:SS`, or `"∞"` if infinite or NaN.
    #[must_use]
    pub fn eta_hms(&self) -> String {
        if !self.eta_secs.is_finite() {
            return "\u{221e}".to_owned(); // ∞
        }
        let total_secs = self.eta_secs as u64;
        let h = total_secs / 3600;
        let m = (total_secs % 3600) / 60;
        let s = total_secs % 60;
        format!("{h:02}:{m:02}:{s:02}")
    }

    /// Progress percentage in [0.0, 100.0].
    #[must_use]
    pub fn progress_pct(&self) -> f64 {
        self.progress_fraction * 100.0
    }

    /// The ETA lower bound (eta - confidence_interval), floored at 0.
    #[must_use]
    pub fn eta_lower_secs(&self) -> f64 {
        (self.eta_secs - self.confidence_interval_secs).max(0.0)
    }

    /// The ETA upper bound (eta + confidence_interval).
    #[must_use]
    pub fn eta_upper_secs(&self) -> f64 {
        self.eta_secs + self.confidence_interval_secs
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_sample(frame: u32, secs: f64) -> CompletionSample {
        CompletionSample::new(WorkerClass::CpuOnly, secs, frame, "cpu-01")
    }

    fn gpu_sample(frame: u32, secs: f64) -> CompletionSample {
        CompletionSample::new(WorkerClass::GpuHighEnd, secs, frame, "gpu-01")
    }

    fn alloc_cpu(n: u32) -> WorkerClassAllocation {
        let mut a = WorkerClassAllocation::new();
        a.set(WorkerClass::CpuOnly, n);
        a
    }

    fn alloc_gpu(n: u32) -> WorkerClassAllocation {
        let mut a = WorkerClassAllocation::new();
        a.set(WorkerClass::GpuHighEnd, n);
        a
    }

    // ── WorkerClass ──────────────────────────────────────────────────────────

    #[test]
    fn test_worker_class_display_names() {
        assert_eq!(WorkerClass::CpuOnly.display_name(), "CPU-Only");
        assert_eq!(WorkerClass::GpuHighEnd.display_name(), "GPU-HighEnd");
        let c = WorkerClass::Custom("MyRig".to_owned());
        assert_eq!(c.display_name(), "MyRig");
    }

    #[test]
    fn test_worker_class_baseline_weights_ordered() {
        // GPU-HighEnd must have the highest baseline weight
        assert!(
            WorkerClass::GpuHighEnd.baseline_weight() > WorkerClass::GpuStandard.baseline_weight()
        );
        assert!(
            WorkerClass::GpuStandard.baseline_weight() > WorkerClass::CpuOnly.baseline_weight()
        );
    }

    // ── CompletionSample ─────────────────────────────────────────────────────

    #[test]
    fn test_completion_sample_duration_clamped() {
        let s = CompletionSample::new(WorkerClass::CpuOnly, -1.0, 1, "w");
        assert!(s.duration_secs >= 1e-3);
    }

    // ── WorkerClassAllocation ────────────────────────────────────────────────

    #[test]
    fn test_allocation_total_workers() {
        let mut a = WorkerClassAllocation::new();
        a.set(WorkerClass::CpuOnly, 4);
        a.set(WorkerClass::GpuHighEnd, 2);
        assert_eq!(a.total_workers(), 6);
    }

    #[test]
    fn test_allocation_set_zero_removes() {
        let mut a = WorkerClassAllocation::new();
        a.set(WorkerClass::CpuOnly, 3);
        a.set(WorkerClass::CpuOnly, 0);
        assert_eq!(a.total_workers(), 0);
    }

    #[test]
    fn test_allocation_add_one() {
        let mut a = WorkerClassAllocation::new();
        a.add_one(WorkerClass::GpuStandard);
        a.add_one(WorkerClass::GpuStandard);
        assert_eq!(a.total_workers(), 2);
    }

    // ── EtaPredictor — basic ─────────────────────────────────────────────────

    #[test]
    fn test_predictor_zero_frames_complete() {
        let predictor = EtaPredictor::new(0);
        assert_eq!(predictor.progress(), 1.0);
        assert_eq!(predictor.remaining_frames(), 0);
    }

    #[test]
    fn test_predictor_progress_fraction() {
        let mut p = EtaPredictor::new(100);
        for i in 0..25 {
            p.ingest(cpu_sample(i, 2.0));
        }
        assert!((p.progress() - 0.25).abs() < 1e-9);
        assert_eq!(p.remaining_frames(), 75);
    }

    #[test]
    fn test_predictor_stall_when_no_allocation() {
        let mut p = EtaPredictor::new(100);
        p.ingest(cpu_sample(1, 2.0));
        let alloc = WorkerClassAllocation::new(); // empty
        let report = p.predict(&alloc);
        assert!(report.is_stalled);
        assert!(report.eta_secs.is_infinite());
    }

    #[test]
    fn test_predictor_eta_finite_with_workers() {
        let mut p = EtaPredictor::new(100);
        // Feed 10 frames, each 2 seconds on CPU
        for i in 0..10 {
            p.ingest(cpu_sample(i, 2.0));
        }
        let alloc = alloc_cpu(2); // 2 CPU workers
        let report = p.predict(&alloc);
        // ETA ≈ 90 remaining frames / (0.5 fps × 2 workers) = 90 s
        assert!(report.eta_secs.is_finite());
        assert!(report.eta_secs > 0.0);
        assert!(!report.is_stalled);
    }

    #[test]
    fn test_predictor_complete_job_eta_zero() {
        let mut p = EtaPredictor::new(10);
        for i in 0..10 {
            p.ingest(cpu_sample(i, 1.0));
        }
        let alloc = alloc_cpu(1);
        let report = p.predict(&alloc);
        assert!(report.is_complete());
        assert_eq!(report.eta_secs, 0.0);
    }

    #[test]
    fn test_predictor_gpu_faster_than_cpu() {
        let mut p_cpu = EtaPredictor::new(1000);
        let mut p_gpu = EtaPredictor::new(1000);

        // CPU: 5 sec/frame; GPU: 0.5 sec/frame
        for i in 0..50 {
            p_cpu.ingest(cpu_sample(i, 5.0));
            p_gpu.ingest(gpu_sample(i, 0.5));
        }

        let alloc = alloc_cpu(1);
        let alloc_g = alloc_gpu(1);

        let eta_cpu = p_cpu.predict(&alloc).eta_secs;
        let eta_gpu = p_gpu.predict(&alloc_g).eta_secs;

        // GPU should predict a much shorter ETA
        assert!(eta_gpu < eta_cpu);
    }

    // ── ProgressReport ───────────────────────────────────────────────────────

    #[test]
    fn test_progress_report_is_complete() {
        let mut p = EtaPredictor::new(5);
        for i in 0..5 {
            p.ingest(cpu_sample(i, 1.0));
        }
        let report = p.predict(&alloc_cpu(1));
        assert!(report.is_complete());
    }

    #[test]
    fn test_progress_report_hms_formatting() {
        let p = EtaPredictor::new(1000);
        // Force EMA toward exactly 1 sec/frame by alpha=1 (use with_alpha)
        let mut p2 = EtaPredictor::with_alpha(1000, 1.0);
        for i in 0..10 {
            p2.ingest(cpu_sample(i, 1.0)); // 1 fps
        }
        let alloc = alloc_cpu(1);
        let report = p2.predict(&alloc);
        // Remaining = 990 frames at 1 fps = 990 s
        let hms = report.eta_hms();
        // Should be non-empty and finite
        assert!(!hms.is_empty());
        assert_ne!(hms, "\u{221e}");
        let _ = p; // suppress unused warning
    }

    #[test]
    fn test_progress_report_hms_infinite() {
        let p = EtaPredictor::new(100);
        let alloc = WorkerClassAllocation::new(); // empty → stall
        let report = p.predict(&alloc);
        assert_eq!(report.eta_hms(), "\u{221e}");
    }

    #[test]
    fn test_progress_report_progress_pct() {
        let mut p = EtaPredictor::new(200);
        for i in 0..50 {
            p.ingest(cpu_sample(i, 1.0));
        }
        let alloc = alloc_cpu(1);
        let report = p.predict(&alloc);
        assert!((report.progress_pct() - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_progress_report_confidence_interval_positive() {
        let mut p = EtaPredictor::new(1000);
        // Introduce variance by varying frame times
        for i in 0..20u32 {
            let t = if i % 2 == 0 { 1.0 } else { 3.0 };
            p.ingest(cpu_sample(i, t));
        }
        let alloc = alloc_cpu(2);
        let report = p.predict(&alloc);
        assert!(report.confidence_interval_secs >= 0.0);
        assert_eq!(
            report.eta_lower_secs(),
            (report.eta_secs - report.confidence_interval_secs).max(0.0)
        );
        assert_eq!(
            report.eta_upper_secs(),
            report.eta_secs + report.confidence_interval_secs
        );
    }

    #[test]
    fn test_predictor_mixed_class_allocation() {
        let mut p = EtaPredictor::new(500);
        // 20 CPU frames at 4 s/frame → ~0.25 fps
        for i in 0..20 {
            p.ingest(cpu_sample(i, 4.0));
        }
        // 20 GPU-HighEnd frames at 0.2 s/frame → 5 fps
        for i in 0..20 {
            p.ingest(gpu_sample(i, 0.2));
        }

        let mut alloc = WorkerClassAllocation::new();
        alloc.set(WorkerClass::CpuOnly, 4);
        alloc.set(WorkerClass::GpuHighEnd, 1);

        let report = p.predict(&alloc);
        // GPU is much faster; mixed ETA should be finite and reasonable
        assert!(report.eta_secs.is_finite());
        assert!(report.aggregate_fps > 0.0);
    }

    #[test]
    fn test_predictor_record_failure_does_not_count_as_completed() {
        let mut p = EtaPredictor::new(10);
        p.ingest(cpu_sample(1, 1.0));
        p.record_failure();
        // Completed = 1, failed = 1, remaining = 9
        assert_eq!(p.completed_frames, 1);
        assert_eq!(p.failed_frames, 1);
        assert_eq!(p.remaining_frames(), 9);
    }

    #[test]
    fn test_predictor_class_throughput_in_report() {
        let mut p = EtaPredictor::new(100);
        for i in 0..5 {
            p.ingest(cpu_sample(i, 2.0)); // 0.5 fps
        }
        let alloc = alloc_cpu(1);
        let report = p.predict(&alloc);
        let cpu_fps = report.class_throughput_fps.get("CPU-Only");
        assert!(cpu_fps.is_some());
        let fps = cpu_fps.copied().unwrap_or(0.0);
        assert!((fps - 0.5).abs() < 0.01);
    }
}
