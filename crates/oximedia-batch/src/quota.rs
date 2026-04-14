//! Resource quota management for batch processing.
//!
//! [`ResourceQuota`] tracks available CPU and memory budgets for concurrent
//! job execution.  Jobs call `consume` before starting (returns `false` if
//! resources are insufficient) and `release` after completion.
//!
//! # Example
//!
//! ```
//! use oximedia_batch::quota::ResourceQuota;
//!
//! let mut quota = ResourceQuota::new(4.0, 8192);
//!
//! // Reserve 2 CPU cores and 2 GB RAM
//! assert!(quota.consume(2.0, 2048));
//! assert!(quota.consume(2.0, 2048));
//!
//! // No more CPU available
//! assert!(!quota.consume(0.1, 0));
//!
//! // Release and try again
//! quota.release(2.0, 2048);
//! assert!(quota.consume(0.1, 0));
//! ```

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// ResourceQuota
// ---------------------------------------------------------------------------

/// Tracks available CPU fraction and memory (MB) for concurrent batch jobs.
#[derive(Debug, Clone)]
pub struct ResourceQuota {
    /// Maximum CPU fraction (e.g. 4.0 = 4 cores).
    max_cpu: f32,
    /// Maximum memory in megabytes.
    max_mem_mb: u64,
    /// Currently consumed CPU.
    used_cpu: f32,
    /// Currently consumed memory in MB.
    used_mem_mb: u64,
}

impl ResourceQuota {
    /// Create a new `ResourceQuota` with the given limits.
    ///
    /// `max_cpu` is in fractional core units (e.g. `4.0` = 4 cores).
    /// `max_mem_mb` is in mebibytes.
    #[must_use]
    pub fn new(max_cpu: f32, max_mem_mb: u64) -> Self {
        Self {
            max_cpu: max_cpu.max(0.0),
            max_mem_mb,
            used_cpu: 0.0,
            used_mem_mb: 0,
        }
    }

    /// Attempt to consume `cpu` cores and `mem_mb` megabytes.
    ///
    /// Returns `true` and updates internal counters when sufficient resources
    /// are available.  Returns `false` without modifying state when the
    /// requested amounts would exceed either limit.
    pub fn consume(&mut self, cpu: f32, mem_mb: u64) -> bool {
        let cpu = cpu.max(0.0);
        let new_cpu = self.used_cpu + cpu;
        let new_mem = self.used_mem_mb.saturating_add(mem_mb);

        if new_cpu > self.max_cpu + f32::EPSILON || new_mem > self.max_mem_mb {
            return false;
        }

        self.used_cpu = new_cpu;
        self.used_mem_mb = new_mem;
        true
    }

    /// Release `cpu` cores and `mem_mb` megabytes back to the pool.
    ///
    /// Values are clamped so the used counts never go below zero.
    pub fn release(&mut self, cpu: f32, mem_mb: u64) {
        let cpu = cpu.max(0.0);
        self.used_cpu = (self.used_cpu - cpu).max(0.0);
        self.used_mem_mb = self.used_mem_mb.saturating_sub(mem_mb);
    }

    /// Available (remaining) CPU fraction.
    #[must_use]
    pub fn available_cpu(&self) -> f32 {
        (self.max_cpu - self.used_cpu).max(0.0)
    }

    /// Available (remaining) memory in megabytes.
    #[must_use]
    pub fn available_mem_mb(&self) -> u64 {
        self.max_mem_mb.saturating_sub(self.used_mem_mb)
    }

    /// Currently consumed CPU.
    #[must_use]
    pub fn used_cpu(&self) -> f32 {
        self.used_cpu
    }

    /// Currently consumed memory in megabytes.
    #[must_use]
    pub fn used_mem_mb(&self) -> u64 {
        self.used_mem_mb
    }

    /// Maximum CPU limit.
    #[must_use]
    pub fn max_cpu(&self) -> f32 {
        self.max_cpu
    }

    /// Maximum memory limit in megabytes.
    #[must_use]
    pub fn max_mem_mb(&self) -> u64 {
        self.max_mem_mb
    }

    /// Returns `true` when neither CPU nor memory is in use.
    #[must_use]
    pub fn is_idle(&self) -> bool {
        self.used_cpu < f32::EPSILON && self.used_mem_mb == 0
    }

    /// Reset all usage to zero.
    pub fn reset(&mut self) {
        self.used_cpu = 0.0;
        self.used_mem_mb = 0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── new ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_new_starts_idle() {
        let q = ResourceQuota::new(4.0, 8192);
        assert!(q.is_idle());
        assert_eq!(q.used_cpu(), 0.0_f32);
        assert_eq!(q.used_mem_mb(), 0);
    }

    #[test]
    fn test_new_clamps_negative_cpu() {
        let q = ResourceQuota::new(-1.0, 1024);
        assert_eq!(q.max_cpu(), 0.0_f32);
    }

    // ── consume ───────────────────────────────────────────────────────────────

    #[test]
    fn test_consume_within_limits_returns_true() {
        let mut q = ResourceQuota::new(4.0, 8192);
        assert!(q.consume(2.0, 2048));
        assert!((q.used_cpu() - 2.0).abs() < 1e-4);
        assert_eq!(q.used_mem_mb(), 2048);
    }

    #[test]
    fn test_consume_exact_limits_returns_true() {
        let mut q = ResourceQuota::new(2.0, 4096);
        assert!(q.consume(2.0, 4096));
        assert!(!q.consume(0.001, 0), "Should fail: CPU exceeded");
    }

    #[test]
    fn test_consume_exceeds_cpu_returns_false() {
        let mut q = ResourceQuota::new(2.0, 8192);
        assert!(!q.consume(3.0, 0));
        assert!(q.is_idle(), "State should not change on failure");
    }

    #[test]
    fn test_consume_exceeds_mem_returns_false() {
        let mut q = ResourceQuota::new(4.0, 1024);
        assert!(!q.consume(0.0, 2048));
        assert!(q.is_idle());
    }

    #[test]
    fn test_consume_negative_cpu_treated_as_zero() {
        let mut q = ResourceQuota::new(1.0, 1024);
        assert!(q.consume(-0.5, 0));
        assert!((q.used_cpu() - 0.0).abs() < 1e-4);
    }

    // ── release ───────────────────────────────────────────────────────────────

    #[test]
    fn test_release_reduces_used() {
        let mut q = ResourceQuota::new(4.0, 8192);
        q.consume(2.0, 2048);
        q.release(1.0, 1024);
        assert!((q.used_cpu() - 1.0).abs() < 1e-4);
        assert_eq!(q.used_mem_mb(), 1024);
    }

    #[test]
    fn test_release_clamps_at_zero() {
        let mut q = ResourceQuota::new(4.0, 8192);
        q.consume(1.0, 512);
        q.release(5.0, 1000); // more than consumed
        assert!((q.used_cpu()).abs() < 1e-4);
        assert_eq!(q.used_mem_mb(), 0);
    }

    #[test]
    fn test_release_restores_availability() {
        let mut q = ResourceQuota::new(2.0, 4096);
        q.consume(2.0, 4096);
        assert!(!q.consume(0.1, 0));
        q.release(2.0, 4096);
        assert!(q.consume(0.1, 0), "Resources freed, should succeed now");
    }

    // ── available_* ──────────────────────────────────────────────────────────

    #[test]
    fn test_available_cpu_decreases_after_consume() {
        let mut q = ResourceQuota::new(4.0, 8192);
        q.consume(1.5, 0);
        assert!((q.available_cpu() - 2.5).abs() < 1e-3);
    }

    #[test]
    fn test_available_mem_decreases_after_consume() {
        let mut q = ResourceQuota::new(4.0, 8192);
        q.consume(0.0, 3000);
        assert_eq!(q.available_mem_mb(), 5192);
    }

    // ── reset ────────────────────────────────────────────────────────────────

    #[test]
    fn test_reset_clears_usage() {
        let mut q = ResourceQuota::new(4.0, 8192);
        q.consume(4.0, 8192);
        q.reset();
        assert!(q.is_idle());
    }
}
