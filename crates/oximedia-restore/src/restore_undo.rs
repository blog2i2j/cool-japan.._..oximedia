#![allow(dead_code)]
//! Per-step rollback capability for restoration chains.
//!
//! When performing multi-step audio restoration it is common to discover that
//! a particular step introduced artefacts or was misconfigured.  Rather than
//! re-processing the entire chain from scratch, this module stores
//! intermediate buffers after each step so that you can *undo* one or more
//! steps cheaply.
//!
//! # Design
//!
//! - [`UndoHistory`] wraps a sequence of snapshots, one per completed step.
//! - Each [`UndoSnapshot`] stores the buffer state **after** the step ran,
//!   together with metadata (step index, step name, timestamp).
//! - You can roll back to any earlier snapshot by index, or pop the most
//!   recent one.
//! - An optional memory budget limits the total size of stored buffers; when
//!   the budget is exceeded the oldest snapshots are evicted.
//!
//! # Example
//!
//! ```
//! use oximedia_restore::restore_undo::*;
//!
//! let mut history = UndoHistory::new(UndoConfig::default());
//! let initial = vec![0.1_f32, 0.2, 0.3];
//! history.push_snapshot("dc_removal", 0, initial.clone());
//! assert_eq!(history.len(), 1);
//!
//! let step2 = vec![0.11, 0.21, 0.31];
//! history.push_snapshot("hum_removal", 1, step2);
//! assert_eq!(history.len(), 2);
//!
//! // Undo last step
//! let prev = history.undo().unwrap();
//! assert_eq!(prev.samples, initial);
//! ```

use std::time::Instant;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the undo history.
#[derive(Debug, Clone)]
pub struct UndoConfig {
    /// Maximum number of snapshots to retain.
    /// When exceeded, the oldest snapshot is evicted.
    pub max_snapshots: usize,
    /// Maximum total memory budget in bytes for all stored buffers.
    /// 0 = unlimited.
    pub max_memory_bytes: usize,
    /// Whether to store stereo snapshots as interleaved pairs.
    pub store_stereo: bool,
}

impl Default for UndoConfig {
    fn default() -> Self {
        Self {
            max_snapshots: 32,
            max_memory_bytes: 0, // unlimited
            store_stereo: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Snapshot
// ---------------------------------------------------------------------------

/// A snapshot of the audio buffer after a restoration step.
#[derive(Debug, Clone)]
pub struct UndoSnapshot {
    /// Human-readable name of the step that produced this snapshot.
    pub step_name: String,
    /// Index of the step in the restoration chain.
    pub step_index: usize,
    /// The mono sample buffer at this point.
    pub samples: Vec<f32>,
    /// Optional right channel for stereo snapshots.
    pub right_channel: Option<Vec<f32>>,
    /// Timestamp when the snapshot was created.
    pub created_at: Instant,
    /// Approximate memory footprint of this snapshot in bytes.
    pub memory_bytes: usize,
}

impl UndoSnapshot {
    /// Create a mono snapshot.
    pub fn mono(step_name: &str, step_index: usize, samples: Vec<f32>) -> Self {
        let memory_bytes = samples.len() * std::mem::size_of::<f32>();
        Self {
            step_name: step_name.to_string(),
            step_index,
            samples,
            right_channel: None,
            created_at: Instant::now(),
            memory_bytes,
        }
    }

    /// Create a stereo snapshot.
    pub fn stereo(step_name: &str, step_index: usize, left: Vec<f32>, right: Vec<f32>) -> Self {
        let memory_bytes = (left.len() + right.len()) * std::mem::size_of::<f32>();
        Self {
            step_name: step_name.to_string(),
            step_index,
            samples: left,
            right_channel: Some(right),
            created_at: Instant::now(),
            memory_bytes,
        }
    }

    /// Returns `true` if this is a stereo snapshot.
    pub fn is_stereo(&self) -> bool {
        self.right_channel.is_some()
    }
}

// ---------------------------------------------------------------------------
// Undo history
// ---------------------------------------------------------------------------

/// Manages a stack of intermediate restoration buffers with rollback support.
#[derive(Debug)]
pub struct UndoHistory {
    config: UndoConfig,
    snapshots: Vec<UndoSnapshot>,
    total_memory: usize,
}

impl UndoHistory {
    /// Create a new empty undo history.
    pub fn new(config: UndoConfig) -> Self {
        Self {
            config,
            snapshots: Vec::new(),
            total_memory: 0,
        }
    }

    /// Return the number of stored snapshots.
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Return `true` if there are no snapshots.
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Return the total memory consumed by all snapshots (bytes).
    pub fn total_memory_bytes(&self) -> usize {
        self.total_memory
    }

    /// Return a reference to the configuration.
    pub fn config(&self) -> &UndoConfig {
        &self.config
    }

    /// Update the configuration.  Does **not** retroactively evict snapshots;
    /// eviction happens on the next [`push_snapshot`](Self::push_snapshot) call.
    pub fn set_config(&mut self, config: UndoConfig) {
        self.config = config;
    }

    /// Push a mono snapshot onto the history.
    pub fn push_snapshot(&mut self, step_name: &str, step_index: usize, samples: Vec<f32>) {
        let snap = UndoSnapshot::mono(step_name, step_index, samples);
        self.push_snapshot_inner(snap);
    }

    /// Push a stereo snapshot onto the history.
    pub fn push_stereo_snapshot(
        &mut self,
        step_name: &str,
        step_index: usize,
        left: Vec<f32>,
        right: Vec<f32>,
    ) {
        let snap = UndoSnapshot::stereo(step_name, step_index, left, right);
        self.push_snapshot_inner(snap);
    }

    fn push_snapshot_inner(&mut self, snap: UndoSnapshot) {
        self.total_memory += snap.memory_bytes;
        self.snapshots.push(snap);
        self.enforce_limits();
    }

    /// Enforce snapshot count and memory budget limits by evicting the oldest
    /// snapshots.
    fn enforce_limits(&mut self) {
        // Evict by count
        while self.snapshots.len() > self.config.max_snapshots {
            if let Some(removed) = self.evict_oldest() {
                self.total_memory = self.total_memory.saturating_sub(removed.memory_bytes);
            }
        }

        // Evict by memory budget
        if self.config.max_memory_bytes > 0 {
            while self.total_memory > self.config.max_memory_bytes && !self.snapshots.is_empty() {
                if let Some(removed) = self.evict_oldest() {
                    self.total_memory = self.total_memory.saturating_sub(removed.memory_bytes);
                }
            }
        }
    }

    fn evict_oldest(&mut self) -> Option<UndoSnapshot> {
        if self.snapshots.is_empty() {
            None
        } else {
            Some(self.snapshots.remove(0))
        }
    }

    /// Undo the most recent step by popping the last snapshot and returning
    /// the one before it (the state to restore to).
    ///
    /// Returns `None` if there are fewer than 2 snapshots (nothing to undo to).
    pub fn undo(&mut self) -> Option<UndoSnapshot> {
        if self.snapshots.len() < 2 {
            // Can't undo if there's only 0 or 1 snapshot
            let removed = self.snapshots.pop()?;
            self.total_memory = self.total_memory.saturating_sub(removed.memory_bytes);
            // Return the removed snapshot so caller knows what was undone,
            // but there's no earlier state
            return Some(removed);
        }

        // Pop the current state
        let current = self.snapshots.pop()?;
        self.total_memory = self.total_memory.saturating_sub(current.memory_bytes);

        // Return a clone of the previous state (now the latest)
        self.snapshots.last().cloned()
    }

    /// Undo to a specific snapshot index, discarding all snapshots after it.
    ///
    /// Returns a clone of the snapshot at `index`, or `None` if `index` is
    /// out of bounds.
    pub fn undo_to(&mut self, index: usize) -> Option<UndoSnapshot> {
        if index >= self.snapshots.len() {
            return None;
        }

        // Remove everything after `index`
        while self.snapshots.len() > index + 1 {
            if let Some(removed) = self.snapshots.pop() {
                self.total_memory = self.total_memory.saturating_sub(removed.memory_bytes);
            }
        }

        self.snapshots.last().cloned()
    }

    /// Peek at the most recent snapshot without removing it.
    pub fn peek(&self) -> Option<&UndoSnapshot> {
        self.snapshots.last()
    }

    /// Peek at a snapshot by index.
    pub fn get(&self, index: usize) -> Option<&UndoSnapshot> {
        self.snapshots.get(index)
    }

    /// Return a list of step names in the history (oldest first).
    pub fn step_names(&self) -> Vec<&str> {
        self.snapshots
            .iter()
            .map(|s| s.step_name.as_str())
            .collect()
    }

    /// Clear all snapshots.
    pub fn clear(&mut self) {
        self.snapshots.clear();
        self.total_memory = 0;
    }

    /// Return the number of undo steps available.
    ///
    /// This is `len() - 1` (you can undo back to the first snapshot) or 0 if
    /// the history is empty.
    pub fn undo_depth(&self) -> usize {
        self.snapshots.len().saturating_sub(1)
    }

    /// Summary of the history for diagnostic display.
    pub fn summary(&self) -> UndoHistorySummary {
        UndoHistorySummary {
            snapshot_count: self.snapshots.len(),
            total_memory_bytes: self.total_memory,
            max_snapshots: self.config.max_snapshots,
            max_memory_bytes: self.config.max_memory_bytes,
            step_names: self.step_names().iter().map(|s| s.to_string()).collect(),
        }
    }
}

/// Summary of undo history state.
#[derive(Debug, Clone)]
pub struct UndoHistorySummary {
    /// Number of stored snapshots.
    pub snapshot_count: usize,
    /// Total memory used by snapshots in bytes.
    pub total_memory_bytes: usize,
    /// Maximum allowed snapshots.
    pub max_snapshots: usize,
    /// Maximum allowed memory (0 = unlimited).
    pub max_memory_bytes: usize,
    /// Step names in order.
    pub step_names: Vec<String>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = UndoConfig::default();
        assert_eq!(cfg.max_snapshots, 32);
        assert_eq!(cfg.max_memory_bytes, 0);
        assert!(!cfg.store_stereo);
    }

    #[test]
    fn test_empty_history() {
        let history = UndoHistory::new(UndoConfig::default());
        assert!(history.is_empty());
        assert_eq!(history.len(), 0);
        assert_eq!(history.undo_depth(), 0);
        assert_eq!(history.total_memory_bytes(), 0);
    }

    #[test]
    fn test_push_and_len() {
        let mut history = UndoHistory::new(UndoConfig::default());
        history.push_snapshot("step_a", 0, vec![1.0, 2.0, 3.0]);
        assert_eq!(history.len(), 1);
        assert!(!history.is_empty());

        history.push_snapshot("step_b", 1, vec![4.0, 5.0, 6.0]);
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_undo_returns_previous_state() {
        let mut history = UndoHistory::new(UndoConfig::default());
        let buf_a = vec![1.0_f32, 2.0, 3.0];
        let buf_b = vec![4.0_f32, 5.0, 6.0];

        history.push_snapshot("step_a", 0, buf_a.clone());
        history.push_snapshot("step_b", 1, buf_b);

        let undone = history.undo().expect("should have previous");
        assert_eq!(undone.samples, buf_a);
        assert_eq!(undone.step_name, "step_a");
        assert_eq!(history.len(), 1);
    }

    #[test]
    fn test_undo_single_snapshot() {
        let mut history = UndoHistory::new(UndoConfig::default());
        history.push_snapshot("only", 0, vec![1.0]);
        let undone = history.undo();
        assert!(undone.is_some());
        assert!(history.is_empty());
    }

    #[test]
    fn test_undo_empty_returns_none() {
        let mut history = UndoHistory::new(UndoConfig::default());
        assert!(history.undo().is_none());
    }

    #[test]
    fn test_undo_to_index() {
        let mut history = UndoHistory::new(UndoConfig::default());
        history.push_snapshot("a", 0, vec![1.0]);
        history.push_snapshot("b", 1, vec![2.0]);
        history.push_snapshot("c", 2, vec![3.0]);
        history.push_snapshot("d", 3, vec![4.0]);

        let restored = history.undo_to(1).expect("valid index");
        assert_eq!(restored.step_name, "b");
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_undo_to_out_of_bounds() {
        let mut history = UndoHistory::new(UndoConfig::default());
        history.push_snapshot("a", 0, vec![1.0]);
        assert!(history.undo_to(5).is_none());
    }

    #[test]
    fn test_peek() {
        let mut history = UndoHistory::new(UndoConfig::default());
        assert!(history.peek().is_none());

        history.push_snapshot("a", 0, vec![1.0]);
        let peeked = history.peek().expect("has snapshot");
        assert_eq!(peeked.step_name, "a");
        assert_eq!(history.len(), 1); // peek doesn't remove
    }

    #[test]
    fn test_get_by_index() {
        let mut history = UndoHistory::new(UndoConfig::default());
        history.push_snapshot("a", 0, vec![1.0]);
        history.push_snapshot("b", 1, vec![2.0]);

        assert_eq!(history.get(0).expect("ok").step_name, "a");
        assert_eq!(history.get(1).expect("ok").step_name, "b");
        assert!(history.get(2).is_none());
    }

    #[test]
    fn test_step_names() {
        let mut history = UndoHistory::new(UndoConfig::default());
        history.push_snapshot("dc_removal", 0, vec![1.0]);
        history.push_snapshot("hum_removal", 1, vec![2.0]);
        history.push_snapshot("noise_gate", 2, vec![3.0]);

        let names = history.step_names();
        assert_eq!(names, vec!["dc_removal", "hum_removal", "noise_gate"]);
    }

    #[test]
    fn test_clear() {
        let mut history = UndoHistory::new(UndoConfig::default());
        history.push_snapshot("a", 0, vec![1.0; 100]);
        history.push_snapshot("b", 1, vec![2.0; 100]);
        assert!(!history.is_empty());

        history.clear();
        assert!(history.is_empty());
        assert_eq!(history.total_memory_bytes(), 0);
    }

    #[test]
    fn test_max_snapshots_eviction() {
        let cfg = UndoConfig {
            max_snapshots: 3,
            ..Default::default()
        };
        let mut history = UndoHistory::new(cfg);
        history.push_snapshot("a", 0, vec![1.0]);
        history.push_snapshot("b", 1, vec![2.0]);
        history.push_snapshot("c", 2, vec![3.0]);
        history.push_snapshot("d", 3, vec![4.0]); // should evict "a"

        assert_eq!(history.len(), 3);
        assert_eq!(history.get(0).expect("ok").step_name, "b");
    }

    #[test]
    fn test_memory_budget_eviction() {
        let cfg = UndoConfig {
            max_snapshots: 100,
            max_memory_bytes: 16, // very small: 4 floats = 16 bytes
            ..Default::default()
        };
        let mut history = UndoHistory::new(cfg);
        history.push_snapshot("a", 0, vec![1.0; 4]); // 16 bytes, fits
        assert_eq!(history.len(), 1);

        history.push_snapshot("b", 1, vec![2.0; 4]); // 32 bytes total, exceeds
                                                     // At least one eviction should have occurred
        assert!(history.total_memory_bytes() <= 16 || history.len() <= 1);
    }

    #[test]
    fn test_stereo_snapshot() {
        let mut history = UndoHistory::new(UndoConfig::default());
        let left = vec![1.0_f32, 2.0];
        let right = vec![3.0_f32, 4.0];
        history.push_stereo_snapshot("stereo_step", 0, left.clone(), right.clone());

        let snap = history.peek().expect("has snapshot");
        assert!(snap.is_stereo());
        assert_eq!(snap.samples, left);
        assert_eq!(snap.right_channel.as_ref().expect("has right"), &right);
        assert_eq!(snap.memory_bytes, 4 * std::mem::size_of::<f32>());
    }

    #[test]
    fn test_undo_depth() {
        let mut history = UndoHistory::new(UndoConfig::default());
        assert_eq!(history.undo_depth(), 0);

        history.push_snapshot("a", 0, vec![1.0]);
        assert_eq!(history.undo_depth(), 0);

        history.push_snapshot("b", 1, vec![2.0]);
        assert_eq!(history.undo_depth(), 1);

        history.push_snapshot("c", 2, vec![3.0]);
        assert_eq!(history.undo_depth(), 2);
    }

    #[test]
    fn test_summary() {
        let mut history = UndoHistory::new(UndoConfig::default());
        history.push_snapshot("dc", 0, vec![1.0; 10]);
        history.push_snapshot("hum", 1, vec![2.0; 10]);

        let summary = history.summary();
        assert_eq!(summary.snapshot_count, 2);
        assert_eq!(summary.step_names, vec!["dc", "hum"]);
        assert!(summary.total_memory_bytes > 0);
    }

    #[test]
    fn test_set_config() {
        let mut history = UndoHistory::new(UndoConfig::default());
        let new_cfg = UndoConfig {
            max_snapshots: 5,
            max_memory_bytes: 1024,
            store_stereo: true,
        };
        history.set_config(new_cfg);
        assert_eq!(history.config().max_snapshots, 5);
        assert_eq!(history.config().max_memory_bytes, 1024);
    }
}
