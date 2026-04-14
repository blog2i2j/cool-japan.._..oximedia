//! Per-pipeline-stage memory usage tracking.
//!
//! This module provides [`MemoryProfiler`] (a distinct type from the
//! allocation-id–based profiler in `memory_profiler`) that tracks
//! [`AllocationEvent`]s per named pipeline stage and derives per-stage
//! [`MemoryProfile`] / [`StageMemoryStats`] summaries.
//!
//! # Example
//!
//! ```
//! use oximedia_profiler::mem_stage_profiler::MemoryProfiler;
//!
//! let mut profiler = MemoryProfiler::new();
//! profiler.record_alloc("decode", 4096, 0);
//! profiler.record_alloc("decode", 4096, 10);
//! profiler.record_free("decode", 4096, 20);
//!
//! let stats = profiler.stage_stats("decode").expect("stage exists");
//! assert_eq!(stats.profile.peak_bytes, 8192);
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// AllocEventType
// ---------------------------------------------------------------------------

/// Kind of memory event recorded for a stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllocEventType {
    /// A new allocation of `bytes` bytes.
    Alloc,
    /// A release of `bytes` bytes.
    Free,
    /// A reallocation from `old_bytes` to `new_bytes`.
    Realloc {
        /// Size of the original allocation being resized.
        old_bytes: usize,
    },
}

// ---------------------------------------------------------------------------
// AllocationEvent
// ---------------------------------------------------------------------------

/// A single memory event attributed to a pipeline stage.
#[derive(Debug, Clone)]
pub struct AllocationEvent {
    /// Name of the pipeline stage that produced this event.
    pub stage: String,
    /// Number of bytes involved (new size for `Alloc`/`Realloc`, freed size for `Free`).
    pub bytes: usize,
    /// Type of memory event.
    pub event_type: AllocEventType,
    /// Timestamp in microseconds since an arbitrary epoch.
    pub timestamp_us: u64,
}

// ---------------------------------------------------------------------------
// MemoryProfile
// ---------------------------------------------------------------------------

/// Aggregated memory statistics for a single pipeline stage.
#[derive(Debug, Clone, Default)]
pub struct MemoryProfile {
    /// Peak live bytes (i.e. maximum value of `current_bytes` over time).
    pub peak_bytes: usize,
    /// Currently live bytes (can be negative if frees exceed allocs, but
    /// represented as `i64` to surface accounting errors gracefully).
    pub current_bytes: i64,
    /// Total bytes ever allocated (Alloc events only, new side of Realloc).
    pub total_allocated: u64,
    /// Total bytes ever freed (Free events only, old side of Realloc).
    pub total_freed: u64,
    /// Number of Alloc events recorded.
    pub allocation_count: u64,
}

impl MemoryProfile {
    /// Apply the net effect of a single `AllocationEvent` to this profile.
    fn apply(&mut self, event: &AllocationEvent) {
        match &event.event_type {
            AllocEventType::Alloc => {
                self.total_allocated = self.total_allocated.saturating_add(event.bytes as u64);
                self.allocation_count += 1;
                self.current_bytes = self.current_bytes.saturating_add(event.bytes as i64);
                // Update peak
                if self.current_bytes > 0
                    && (self.current_bytes as usize) > self.peak_bytes
                {
                    self.peak_bytes = self.current_bytes as usize;
                }
            }
            AllocEventType::Free => {
                self.total_freed = self.total_freed.saturating_add(event.bytes as u64);
                self.current_bytes = self.current_bytes.saturating_sub(event.bytes as i64);
            }
            AllocEventType::Realloc { old_bytes } => {
                // Net effect: free `old_bytes`, alloc `event.bytes`
                let net = (event.bytes as i64).saturating_sub(*old_bytes as i64);
                self.total_allocated = self.total_allocated.saturating_add(event.bytes as u64);
                self.total_freed = self.total_freed.saturating_add(*old_bytes as u64);
                self.allocation_count += 1;
                self.current_bytes = self.current_bytes.saturating_add(net);
                if self.current_bytes > 0
                    && (self.current_bytes as usize) > self.peak_bytes
                {
                    self.peak_bytes = self.current_bytes as usize;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// StageMemoryStats
// ---------------------------------------------------------------------------

/// Per-stage aggregated memory statistics.
#[derive(Debug, Clone)]
pub struct StageMemoryStats {
    /// Name of the pipeline stage.
    pub stage: String,
    /// Aggregated memory profile for this stage.
    pub profile: MemoryProfile,
}

// ---------------------------------------------------------------------------
// MemoryProfiler
// ---------------------------------------------------------------------------

/// Tracks memory events per named pipeline stage.
#[derive(Debug, Default)]
pub struct MemoryProfiler {
    /// Per-stage running profiles.
    profiles: HashMap<String, MemoryProfile>,
    /// Full event log (useful for auditing / replay).
    events: Vec<AllocationEvent>,
}

impl MemoryProfiler {
    /// Create a new, empty profiler.
    pub fn new() -> Self {
        Self::default()
    }

    // ------------------------------------------------------------------
    // Recording helpers
    // ------------------------------------------------------------------

    fn push_event(&mut self, stage: &str, bytes: usize, event_type: AllocEventType, ts: u64) {
        let event = AllocationEvent {
            stage: stage.to_owned(),
            bytes,
            event_type,
            timestamp_us: ts,
        };
        self.profiles
            .entry(stage.to_owned())
            .or_default()
            .apply(&event);
        self.events.push(event);
    }

    /// Record a memory allocation for `stage`.
    pub fn record_alloc(&mut self, stage: &str, bytes: usize, timestamp_us: u64) {
        self.push_event(stage, bytes, AllocEventType::Alloc, timestamp_us);
    }

    /// Record a memory free for `stage`.
    pub fn record_free(&mut self, stage: &str, bytes: usize, timestamp_us: u64) {
        self.push_event(stage, bytes, AllocEventType::Free, timestamp_us);
    }

    /// Record a reallocation for `stage` (from `old_bytes` to `new_bytes`).
    pub fn record_realloc(
        &mut self,
        stage: &str,
        old_bytes: usize,
        new_bytes: usize,
        timestamp_us: u64,
    ) {
        self.push_event(
            stage,
            new_bytes,
            AllocEventType::Realloc { old_bytes },
            timestamp_us,
        );
    }

    // ------------------------------------------------------------------
    // Query helpers
    // ------------------------------------------------------------------

    /// Returns per-stage stats for the named stage, or `None` if it has never
    /// been seen.
    pub fn stage_stats(&self, stage: &str) -> Option<StageMemoryStats> {
        self.profiles.get(stage).map(|p| StageMemoryStats {
            stage: stage.to_owned(),
            profile: p.clone(),
        })
    }

    /// Returns the top `n` stages by `peak_bytes`, sorted descending.
    pub fn top_consumers(&self, n: usize) -> Vec<StageMemoryStats> {
        let mut stats: Vec<StageMemoryStats> = self
            .profiles
            .iter()
            .map(|(name, profile)| StageMemoryStats {
                stage: name.clone(),
                profile: profile.clone(),
            })
            .collect();
        stats.sort_by(|a, b| b.profile.peak_bytes.cmp(&a.profile.peak_bytes));
        stats.truncate(n);
        stats
    }

    /// Returns a `MemoryProfile` that aggregates every stage together.
    pub fn total_profile(&self) -> MemoryProfile {
        let mut total = MemoryProfile::default();
        for event in &self.events {
            total.apply(event);
        }
        total
    }

    /// All recorded events in insertion order.
    pub fn events(&self) -> &[AllocationEvent] {
        &self.events
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc_increases_current_bytes() {
        let mut p = MemoryProfiler::new();
        p.record_alloc("decode", 1024, 0);
        p.record_alloc("decode", 512, 10);
        let stats = p.stage_stats("decode").expect("stage exists");
        assert_eq!(stats.profile.current_bytes, 1536);
    }

    #[test]
    fn test_free_decreases_current_bytes() {
        let mut p = MemoryProfiler::new();
        p.record_alloc("encode", 2048, 0);
        p.record_free("encode", 2048, 5);
        let stats = p.stage_stats("encode").expect("stage exists");
        assert_eq!(stats.profile.current_bytes, 0);
    }

    #[test]
    fn test_peak_tracked_correctly() {
        let mut p = MemoryProfiler::new();
        p.record_alloc("stage", 4096, 0);
        p.record_alloc("stage", 4096, 1); // peak = 8192
        p.record_free("stage", 4096, 2);  // current drops to 4096, peak stays
        let stats = p.stage_stats("stage").expect("stage exists");
        assert_eq!(stats.profile.peak_bytes, 8192);
        assert_eq!(stats.profile.current_bytes, 4096);
    }

    #[test]
    fn test_realloc_net_effect_grow() {
        let mut p = MemoryProfiler::new();
        p.record_alloc("buf", 1000, 0);
        p.record_realloc("buf", 1000, 2000, 5); // grow: net +1000
        let stats = p.stage_stats("buf").expect("stage exists");
        assert_eq!(stats.profile.current_bytes, 2000);
    }

    #[test]
    fn test_realloc_net_effect_shrink() {
        let mut p = MemoryProfiler::new();
        p.record_alloc("buf", 4000, 0);
        p.record_realloc("buf", 4000, 1000, 5); // shrink: net -3000
        let stats = p.stage_stats("buf").expect("stage exists");
        assert_eq!(stats.profile.current_bytes, 1000);
    }

    #[test]
    fn test_top_consumers_ordering() {
        let mut p = MemoryProfiler::new();
        p.record_alloc("small", 100, 0);
        p.record_alloc("large", 9000, 0);
        p.record_alloc("medium", 500, 0);
        let top2 = p.top_consumers(2);
        assert_eq!(top2[0].stage, "large");
        assert_eq!(top2[1].stage, "medium");
    }

    #[test]
    fn test_top_consumers_respects_n() {
        let mut p = MemoryProfiler::new();
        for i in 0..5u64 {
            p.record_alloc(&format!("stage_{i}"), (i + 1) as usize * 100, 0);
        }
        assert_eq!(p.top_consumers(3).len(), 3);
    }

    #[test]
    fn test_stage_stats_unknown_stage_returns_none() {
        let p = MemoryProfiler::new();
        assert!(p.stage_stats("nonexistent").is_none());
    }

    #[test]
    fn test_total_profile_aggregates_all_stages() {
        let mut p = MemoryProfiler::new();
        p.record_alloc("a", 1000, 0);
        p.record_alloc("b", 2000, 0);
        let total = p.total_profile();
        assert_eq!(total.total_allocated, 3000);
        assert_eq!(total.allocation_count, 2);
    }

    #[test]
    fn test_allocation_count_increments() {
        let mut p = MemoryProfiler::new();
        p.record_alloc("s", 100, 0);
        p.record_alloc("s", 200, 1);
        let stats = p.stage_stats("s").expect("stage exists");
        assert_eq!(stats.profile.allocation_count, 2);
    }
}
