//! Multi-GPU load balancing with automatic frame distribution.
//!
//! This module provides a `MultiGpuScheduler` that distributes frames across
//! all available GPU devices, performing automatic load balancing based on
//! measured per-device throughput and real-time queue depth.
//!
//! # Architecture
//!
//! ```text
//!                       ┌───────────────────────┐
//!                       │  MultiGpuScheduler    │
//!                       │  ─────────────────    │
//!                       │  • device pool        │
//!                       │  • load balancer      │
//!                       │  • frame dispatcher   │
//!                       └──────────┬────────────┘
//!               ┌──────────────────┼────────────────────┐
//!         ┌─────▼─────┐     ┌──────▼──────┐     ┌───────▼──────┐
//!         │  GPU 0    │     │   GPU 1     │     │   GPU n …    │
//!         │  worker   │     │   worker    │     │   worker     │
//!         └───────────┘     └─────────────┘     └──────────────┘
//! ```
//!
//! # Load-Balancing Strategies
//!
//! | Strategy           | Description                                            |
//! |-------------------|--------------------------------------------------------|
//! | `RoundRobin`       | Distribute frames in strict order across devices.      |
//! | `LeastLoaded`      | Always assign to the device with the fewest queued frames. |
//! | `WeightedCapacity` | Assign proportionally to a static device-weight table. |
//! | `AdaptiveThroughput` | Track measured throughput and route to fastest device. |
//!
//! # Status
//!
//! GPU command dispatch is a stub (returns CPU-only results).  The scheduling
//! logic and statistics are fully functional.

use crate::{GpuDevice, GpuError, Result};
use parking_lot::Mutex;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// Load-balancing strategies
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy used by [`MultiGpuScheduler`] to assign work to devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalanceStrategy {
    /// Strict round-robin assignment across all available devices.
    RoundRobin,
    /// Always assign to the device with the smallest pending queue depth.
    LeastLoaded,
    /// Assign proportionally to each device's `weight` in [`DeviceSlot`].
    WeightedCapacity,
    /// Dynamically measure throughput and prefer the fastest device.
    AdaptiveThroughput,
}

impl Default for LoadBalanceStrategy {
    fn default() -> Self {
        Self::LeastLoaded
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-device statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Runtime statistics for a single device slot.
#[derive(Debug, Clone, Default)]
pub struct DeviceStats {
    /// Total frames dispatched to this device.
    pub frames_dispatched: u64,
    /// Total frames completed (successfully processed).
    pub frames_completed: u64,
    /// Total frames that failed.
    pub frames_failed: u64,
    /// Exponential moving-average throughput (frames / second).
    pub ema_throughput_fps: f64,
    /// Current pending queue depth (dispatched but not yet completed).
    pub queue_depth: u64,
}

impl DeviceStats {
    /// Update the EMA throughput given a new measurement (`fps`).
    pub fn update_ema(&mut self, fps: f64) {
        const ALPHA: f64 = 0.1;
        if self.ema_throughput_fps == 0.0 {
            self.ema_throughput_fps = fps;
        } else {
            self.ema_throughput_fps = ALPHA * fps + (1.0 - ALPHA) * self.ema_throughput_fps;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Device slot
// ─────────────────────────────────────────────────────────────────────────────

/// A device slot held by the multi-GPU scheduler.
pub struct DeviceSlot {
    /// The GPU device.
    pub device: Arc<GpuDevice>,
    /// Static capacity weight (used by [`LoadBalanceStrategy::WeightedCapacity`]).
    pub weight: f32,
    /// Device-level statistics (protected by a mutex for multi-threaded access).
    pub stats: Mutex<DeviceStats>,
    /// Unique index assigned by the scheduler.
    pub index: usize,
}

impl DeviceSlot {
    /// Create a new device slot.
    #[must_use]
    pub fn new(device: Arc<GpuDevice>, index: usize, weight: f32) -> Self {
        Self {
            device,
            weight: weight.max(0.01),
            stats: Mutex::new(DeviceStats::default()),
            index,
        }
    }

    /// Record a dispatched frame.
    pub fn on_dispatch(&self) {
        let mut s = self.stats.lock();
        s.frames_dispatched += 1;
        s.queue_depth += 1;
    }

    /// Record a completed frame with the measured latency in seconds.
    pub fn on_complete(&self, latency_secs: f64) {
        let mut s = self.stats.lock();
        s.frames_completed += 1;
        s.queue_depth = s.queue_depth.saturating_sub(1);
        if latency_secs > 0.0 {
            s.update_ema(1.0 / latency_secs);
        }
    }

    /// Record a failed frame.
    pub fn on_failure(&self) {
        let mut s = self.stats.lock();
        s.frames_failed += 1;
        s.queue_depth = s.queue_depth.saturating_sub(1);
    }

    /// Current queue depth (lock-free snapshot).
    #[must_use]
    pub fn queue_depth(&self) -> u64 {
        self.stats.lock().queue_depth
    }

    /// Current EMA throughput (lock-free snapshot).
    #[must_use]
    pub fn ema_throughput(&self) -> f64 {
        self.stats.lock().ema_throughput_fps
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MultiGpuScheduler
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-GPU frame scheduler with configurable load-balancing.
///
/// # Thread Safety
///
/// `MultiGpuScheduler` is `Send + Sync` and can be shared across threads via
/// `Arc`.  The internal round-robin counter is protected by a [`Mutex`].
pub struct MultiGpuScheduler {
    slots: Vec<DeviceSlot>,
    strategy: LoadBalanceStrategy,
    rr_counter: Mutex<usize>,
}

impl MultiGpuScheduler {
    /// Create a new scheduler from a list of `(device, weight)` pairs.
    ///
    /// # Errors
    ///
    /// Returns `GpuError::NotSupported` if `devices` is empty.
    pub fn new(devices: Vec<(Arc<GpuDevice>, f32)>, strategy: LoadBalanceStrategy) -> Result<Self> {
        if devices.is_empty() {
            return Err(GpuError::NotSupported(
                "MultiGpuScheduler requires at least one device".to_string(),
            ));
        }
        let slots = devices
            .into_iter()
            .enumerate()
            .map(|(i, (dev, w))| DeviceSlot::new(dev, i, w))
            .collect();
        Ok(Self {
            slots,
            strategy,
            rr_counter: Mutex::new(0),
        })
    }

    /// Create a scheduler from a list of devices with equal weights using the
    /// default `LeastLoaded` strategy.
    ///
    /// # Errors
    ///
    /// Returns an error if `devices` is empty.
    pub fn equal_weight(devices: Vec<Arc<GpuDevice>>) -> Result<Self> {
        Self::new(
            devices.into_iter().map(|d| (d, 1.0)).collect(),
            LoadBalanceStrategy::default(),
        )
    }

    /// Number of devices managed by this scheduler.
    #[must_use]
    pub fn device_count(&self) -> usize {
        self.slots.len()
    }

    /// Select the best device slot index for the next frame according to the
    /// current strategy.
    #[must_use]
    pub fn select_device(&self) -> usize {
        match self.strategy {
            LoadBalanceStrategy::RoundRobin => self.select_round_robin(),
            LoadBalanceStrategy::LeastLoaded => self.select_least_loaded(),
            LoadBalanceStrategy::WeightedCapacity => self.select_weighted(),
            LoadBalanceStrategy::AdaptiveThroughput => self.select_adaptive(),
        }
    }

    /// Dispatch a frame to the best available device.
    ///
    /// Returns the index of the selected device slot.
    ///
    /// The `work_fn` closure receives the selected `GpuDevice` and performs
    /// the actual GPU work.  On success the measured latency (in seconds) is
    /// reported via `on_complete`; on failure `on_failure` is called.
    pub fn dispatch<F, T>(&self, work_fn: F) -> Result<(T, usize)>
    where
        F: FnOnce(&GpuDevice) -> Result<T>,
    {
        let slot_idx = self.select_device();
        let slot = &self.slots[slot_idx];

        slot.on_dispatch();

        let start = std::time::Instant::now();
        match work_fn(&slot.device) {
            Ok(result) => {
                let elapsed = start.elapsed().as_secs_f64();
                slot.on_complete(elapsed);
                Ok((result, slot_idx))
            }
            Err(e) => {
                slot.on_failure();
                Err(e)
            }
        }
    }

    /// Get a snapshot of per-device statistics.
    #[must_use]
    pub fn device_stats(&self) -> Vec<DeviceStats> {
        self.slots.iter().map(|s| s.stats.lock().clone()).collect()
    }

    /// Total frames dispatched across all devices.
    #[must_use]
    pub fn total_dispatched(&self) -> u64 {
        self.slots
            .iter()
            .map(|s| s.stats.lock().frames_dispatched)
            .sum()
    }

    /// Total frames completed (successfully) across all devices.
    #[must_use]
    pub fn total_completed(&self) -> u64 {
        self.slots
            .iter()
            .map(|s| s.stats.lock().frames_completed)
            .sum()
    }

    /// Get a reference to the device slot at `index`.
    ///
    /// Returns `None` if the index is out of range.
    #[must_use]
    pub fn slot(&self, index: usize) -> Option<&DeviceSlot> {
        self.slots.get(index)
    }

    // ── Selection algorithms ─────────────────────────────────────────────────

    fn select_round_robin(&self) -> usize {
        let mut counter = self.rr_counter.lock();
        let idx = *counter % self.slots.len();
        *counter = counter.wrapping_add(1);
        idx
    }

    fn select_least_loaded(&self) -> usize {
        self.slots
            .iter()
            .enumerate()
            .min_by_key(|(_, s)| s.queue_depth())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn select_weighted(&self) -> usize {
        // Weighted random selection: pick a threshold in [0, total_weight) and
        // walk the slots accumulating weights.
        let total_weight: f32 = self.slots.iter().map(|s| s.weight).sum();
        if total_weight <= 0.0 {
            return 0;
        }

        // Use a simple deterministic approximation (no randomness required for
        // deterministic scheduling): find the slot whose cumulative weight
        // share is largest relative to its queue depth.
        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;
        for (i, slot) in self.slots.iter().enumerate() {
            let depth = slot.queue_depth() as f32 + 1.0;
            let score = slot.weight / (total_weight * depth);
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }
        best_idx
    }

    fn select_adaptive(&self) -> usize {
        // Prefer devices with the highest EMA throughput; break ties by queue depth.
        self.slots
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let score_a = a.ema_throughput() / (a.queue_depth() as f64 + 1.0);
                let score_b = b.ema_throughput() / (b.queue_depth() as f64 + 1.0);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Frame distribution helper
// ─────────────────────────────────────────────────────────────────────────────

/// High-level helper that distributes a batch of frames across devices.
///
/// `frames` is a slice of input payloads; `work_fn` is called once per frame
/// with the selected device and the frame payload.
///
/// Returns a `Vec<Result<T>>` in the same order as `frames`.
pub fn distribute_frames<P, T, F>(
    scheduler: &MultiGpuScheduler,
    frames: &[P],
    work_fn: F,
) -> Vec<Result<T>>
where
    P: Send + Sync,
    T: Send,
    F: Fn(&GpuDevice, &P) -> Result<T> + Send + Sync,
{
    frames
        .iter()
        .map(|frame| {
            scheduler
                .dispatch(|dev| work_fn(dev, frame))
                .map(|(result, _)| result)
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a lightweight mock scheduler that uses CPU-fallback devices.
    fn make_scheduler(n: usize, strategy: LoadBalanceStrategy) -> Option<MultiGpuScheduler> {
        let mut devices = Vec::with_capacity(n);
        for _ in 0..n {
            let dev = GpuDevice::new_fallback().ok()?;
            devices.push((Arc::new(dev), 1.0));
        }
        MultiGpuScheduler::new(devices, strategy).ok()
    }

    #[test]
    fn test_empty_device_list_is_error() {
        let result = MultiGpuScheduler::new(vec![], LoadBalanceStrategy::RoundRobin);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_device_always_selected() {
        let Some(sched) = make_scheduler(1, LoadBalanceStrategy::RoundRobin) else {
            return;
        };
        for _ in 0..5 {
            assert_eq!(sched.select_device(), 0);
        }
    }

    #[test]
    fn test_round_robin_cycles() {
        let Some(sched) = make_scheduler(3, LoadBalanceStrategy::RoundRobin) else {
            return;
        };
        let selected: Vec<usize> = (0..6).map(|_| sched.select_device()).collect();
        assert_eq!(selected, vec![0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn test_least_loaded_prefers_idle() {
        let Some(sched) = make_scheduler(3, LoadBalanceStrategy::LeastLoaded) else {
            return;
        };
        // Manually add queue depth to slots 0 and 1.
        sched.slots[0].on_dispatch();
        sched.slots[0].on_dispatch();
        sched.slots[1].on_dispatch();
        // Slot 2 has depth 0 — should be selected.
        assert_eq!(sched.select_device(), 2);
    }

    #[test]
    fn test_dispatch_records_stats() {
        let Some(sched) = make_scheduler(1, LoadBalanceStrategy::RoundRobin) else {
            return;
        };
        let _ = sched.dispatch(|_dev| Ok::<u32, crate::GpuError>(42));
        assert_eq!(sched.total_dispatched(), 1);
        assert_eq!(sched.total_completed(), 1);
    }

    #[test]
    fn test_dispatch_failure_recorded() {
        let Some(sched) = make_scheduler(1, LoadBalanceStrategy::RoundRobin) else {
            return;
        };
        let _ = sched.dispatch(|_dev| {
            Err::<u32, crate::GpuError>(GpuError::NotSupported("test".to_string()))
        });
        let stats = sched.device_stats();
        assert_eq!(stats[0].frames_failed, 1);
        assert_eq!(stats[0].queue_depth, 0);
    }

    #[test]
    fn test_device_count() {
        let Some(sched) = make_scheduler(4, LoadBalanceStrategy::LeastLoaded) else {
            return;
        };
        assert_eq!(sched.device_count(), 4);
    }

    #[test]
    fn test_total_dispatched_sum() {
        let Some(sched) = make_scheduler(3, LoadBalanceStrategy::RoundRobin) else {
            return;
        };
        for _ in 0..9 {
            let _ = sched.dispatch(|_| Ok::<(), _>(()));
        }
        assert_eq!(sched.total_dispatched(), 9);
    }

    #[test]
    fn test_weighted_selects_highest_weight() {
        // Give slot 2 a much higher weight.
        let mk = || GpuDevice::new_fallback().ok().map(Arc::new);
        let (Some(dev0), Some(dev1), Some(dev2)) = (mk(), mk(), mk()) else {
            return;
        };
        let devices: Vec<(Arc<GpuDevice>, f32)> = vec![(dev0, 1.0), (dev1, 1.0), (dev2, 10.0)];
        let Ok(sched) = MultiGpuScheduler::new(devices, LoadBalanceStrategy::WeightedCapacity)
        else {
            return;
        };
        // Without any queue depth, the highest weight should win.
        assert_eq!(sched.select_device(), 2);
    }

    #[test]
    fn test_adaptive_prefers_high_throughput() {
        let Some(sched) = make_scheduler(3, LoadBalanceStrategy::AdaptiveThroughput) else {
            return;
        };
        // Simulate device 1 completing frames quickly.
        sched.slots[1].on_dispatch();
        sched.slots[1].on_complete(0.001); // 1000 fps
        sched.slots[0].on_dispatch();
        sched.slots[0].on_complete(0.1); // 10 fps
                                         // Device 1 should be selected next.
        assert_eq!(sched.select_device(), 1);
    }

    #[test]
    fn test_distribute_frames_returns_results_in_order() {
        let Some(sched) = make_scheduler(2, LoadBalanceStrategy::RoundRobin) else {
            return;
        };
        let frames = vec![1u32, 2, 3, 4, 5, 6];
        let results = distribute_frames(&sched, &frames, |_dev, &frame| Ok(frame * 2));
        let values: Vec<u32> = results
            .into_iter()
            .map(|r| r.expect("frame result"))
            .collect();
        assert_eq!(values, vec![2, 4, 6, 8, 10, 12]);
    }

    #[test]
    fn test_device_stats_snapshot() {
        let Some(sched) = make_scheduler(2, LoadBalanceStrategy::RoundRobin) else {
            return;
        };
        let _ = sched.dispatch(|_| Ok::<(), _>(()));
        let _ = sched.dispatch(|_| Ok::<(), _>(()));
        let stats = sched.device_stats();
        assert_eq!(stats.len(), 2);
        // Round-robin: slot 0 gets frame 0, slot 1 gets frame 1.
        assert_eq!(stats[0].frames_dispatched, 1);
        assert_eq!(stats[1].frames_dispatched, 1);
    }

    #[test]
    fn test_device_ema_update() {
        let mut s = DeviceStats::default();
        s.update_ema(100.0);
        assert!((s.ema_throughput_fps - 100.0).abs() < 1e-6);
        s.update_ema(50.0);
        // EMA with alpha=0.1: 0.1*50 + 0.9*100 = 95
        assert!((s.ema_throughput_fps - 95.0).abs() < 1e-6);
    }
}
