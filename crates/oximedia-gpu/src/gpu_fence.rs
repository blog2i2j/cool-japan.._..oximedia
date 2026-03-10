//! GPU fence (synchronization primitive) management for `oximedia-gpu`.
//!
//! Provides a CPU-side model for GPU fences used to track command completion,
//! plus a pool that hands out and collects fences for reuse.

#![allow(dead_code)]

use std::time::{Duration, Instant};

/// Lifecycle state of a GPU fence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FenceStatus {
    /// Fence has been created but not yet signalled.
    Pending,
    /// GPU has finished executing up to the signalled point.
    Signalled,
    /// Fence was reset after being signalled; can be reused.
    Reset,
    /// Fence timed out while waiting.
    TimedOut,
}

impl FenceStatus {
    /// Returns `true` when execution up to the fence point has completed.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        matches!(self, Self::Signalled)
    }

    /// Returns `true` when the fence can be submitted again.
    #[must_use]
    pub fn is_reusable(&self) -> bool {
        matches!(self, Self::Reset | Self::TimedOut)
    }
}

/// A logical GPU fence (CPU-side descriptor).
///
/// In real GPU code this would wrap a `wgpu::QuerySet` or a Vulkan `VkFence`;
/// here it tracks status in CPU memory.
#[derive(Debug, Clone)]
pub struct GpuFence {
    /// Unique identifier.
    pub id: u64,
    /// Current status.
    pub status: FenceStatus,
    /// Optional debug label.
    pub label: Option<String>,
    /// Simulated "signal time" used in wait operations.
    signal_time: Option<Instant>,
    /// Simulated GPU latency (how long after signal we consider it "done").
    simulated_latency: Duration,
}

impl GpuFence {
    /// Creates a new fence in `Pending` state.
    #[must_use]
    pub fn new(id: u64) -> Self {
        Self {
            id,
            status: FenceStatus::Pending,
            label: None,
            signal_time: None,
            simulated_latency: Duration::from_millis(1),
        }
    }

    /// Attaches a debug label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Signals the fence, marking GPU work as submitted.
    ///
    /// In a real implementation this would record the fence into a command
    /// buffer; here we just record the signal time.
    pub fn signal(&mut self) {
        self.signal_time = Some(Instant::now());
        self.status = FenceStatus::Signalled;
    }

    /// Returns `true` when the fence has been signalled.
    #[must_use]
    pub fn is_signalled(&self) -> bool {
        self.status.is_complete()
    }

    /// Resets the fence so it can be reused.
    pub fn reset(&mut self) {
        self.status = FenceStatus::Reset;
        self.signal_time = None;
    }

    /// Blocks (simulated) until the fence is signalled or `timeout_ms`
    /// milliseconds elapse.
    ///
    /// Returns `true` if the fence was already signalled (no real blocking in
    /// this CPU-only simulation).
    #[allow(clippy::cast_precision_loss)]
    pub fn wait_timeout_ms(&mut self, timeout_ms: u64) -> bool {
        match self.status {
            FenceStatus::Signalled => true,
            FenceStatus::Pending => {
                if let Some(t) = self.signal_time {
                    let elapsed = t.elapsed();
                    if elapsed >= self.simulated_latency {
                        self.status = FenceStatus::Signalled;
                        return true;
                    }
                }
                // Check timeout
                let _ = timeout_ms; // In simulation we never actually sleep
                self.status = FenceStatus::TimedOut;
                false
            }
            FenceStatus::Reset | FenceStatus::TimedOut => false,
        }
    }

    /// Returns the simulated latency of this fence.
    #[must_use]
    pub fn simulated_latency(&self) -> Duration {
        self.simulated_latency
    }
}

/// A pool that manages a collection of [`GpuFence`] objects.
#[derive(Debug, Default)]
pub struct GpuFencePool {
    next_id: u64,
    active: Vec<GpuFence>,
    free_list: Vec<GpuFence>,
}

impl GpuFencePool {
    /// Creates a new, empty fence pool.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates or recycles a fence and returns it in `Pending` state.
    pub fn create_fence(&mut self) -> GpuFence {
        if let Some(mut f) = self.free_list.pop() {
            f.reset();
            f.status = FenceStatus::Pending;
            self.active.push(f.clone());
            return f;
        }
        let id = self.next_id;
        self.next_id += 1;
        let fence = GpuFence::new(id);
        self.active.push(fence.clone());
        fence
    }

    /// Returns a fence to the pool.
    pub fn return_fence(&mut self, fence: GpuFence) {
        self.active.retain(|f| f.id != fence.id);
        self.free_list.push(fence);
    }

    /// Returns the number of active (in-use) fences.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Returns the number of completed (signalled) fences among active ones.
    #[must_use]
    pub fn completed_count(&self) -> usize {
        self.active.iter().filter(|f| f.is_signalled()).count()
    }

    /// Returns the total number of fences ever created by this pool.
    #[must_use]
    pub fn total_created(&self) -> u64 {
        self.next_id
    }

    /// Returns a reference to all currently active fences.
    #[must_use]
    pub fn active_fences(&self) -> &[GpuFence] {
        &self.active
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fence_status_is_complete_signalled() {
        assert!(FenceStatus::Signalled.is_complete());
    }

    #[test]
    fn test_fence_status_is_complete_pending_false() {
        assert!(!FenceStatus::Pending.is_complete());
    }

    #[test]
    fn test_fence_status_is_reusable_reset() {
        assert!(FenceStatus::Reset.is_reusable());
    }

    #[test]
    fn test_fence_status_is_reusable_signalled_false() {
        assert!(!FenceStatus::Signalled.is_reusable());
    }

    #[test]
    fn test_gpu_fence_new_pending() {
        let f = GpuFence::new(0);
        assert_eq!(f.status, FenceStatus::Pending);
        assert!(!f.is_signalled());
    }

    #[test]
    fn test_gpu_fence_signal_sets_status() {
        let mut f = GpuFence::new(1);
        f.signal();
        assert!(f.is_signalled());
        assert_eq!(f.status, FenceStatus::Signalled);
    }

    #[test]
    fn test_gpu_fence_reset_clears_signal() {
        let mut f = GpuFence::new(2);
        f.signal();
        f.reset();
        assert_eq!(f.status, FenceStatus::Reset);
        assert!(f.signal_time.is_none());
    }

    #[test]
    fn test_gpu_fence_wait_when_signalled_returns_true() {
        let mut f = GpuFence::new(3);
        f.signal();
        assert!(f.wait_timeout_ms(100));
    }

    #[test]
    fn test_gpu_fence_wait_timeout_sets_timed_out() {
        let mut f = GpuFence::new(4);
        // Pending and no signal_time set → timeout
        let result = f.wait_timeout_ms(0);
        assert!(!result);
        assert_eq!(f.status, FenceStatus::TimedOut);
    }

    #[test]
    fn test_gpu_fence_with_label() {
        let f = GpuFence::new(5).with_label("frame_complete");
        assert_eq!(f.label.as_deref(), Some("frame_complete"));
    }

    #[test]
    fn test_pool_create_fence_pending() {
        let mut pool = GpuFencePool::new();
        let f = pool.create_fence();
        assert_eq!(f.status, FenceStatus::Pending);
    }

    #[test]
    fn test_pool_active_count_increments() {
        let mut pool = GpuFencePool::new();
        pool.create_fence();
        pool.create_fence();
        assert_eq!(pool.active_count(), 2);
    }

    #[test]
    fn test_pool_completed_count_after_signal() {
        let mut pool = GpuFencePool::new();
        let mut f = pool.create_fence();
        f.signal();
        // Update pool's internal copy by re-inserting (pool holds clone)
        // completed_count inspects pool.active; signal happened on detached copy.
        // This tests that completed_count works on in-pool signalled fences.
        pool.active
            .iter_mut()
            .find(|x| x.id == f.id)
            .expect("operation should succeed in test")
            .signal();
        assert_eq!(pool.completed_count(), 1);
    }

    #[test]
    fn test_pool_return_fence_moves_to_free_list() {
        let mut pool = GpuFencePool::new();
        let f = pool.create_fence();
        pool.return_fence(f);
        assert_eq!(pool.active_count(), 0);
    }

    #[test]
    fn test_pool_total_created_monotonic() {
        let mut pool = GpuFencePool::new();
        pool.create_fence();
        pool.create_fence();
        assert_eq!(pool.total_created(), 2);
    }
}
