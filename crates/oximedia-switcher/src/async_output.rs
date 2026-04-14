//! Async, non-blocking output routing for multiviewer and record paths.
//!
//! Wraps the existing synchronous `OutputMatrix` (from `output_routing`) inside
//! a `tokio::sync::RwLock` so that read-heavy multiviewer consumers do not
//! block each other, and a rare write (re-routing an output) does not stall
//! the frame loop.
//!
//! Only compiled on non-WASM targets (same as the tokio dependency).

#![cfg(not(target_arch = "wasm32"))]

use std::sync::Arc;
use tokio::sync::RwLock;

use crate::output_routing::{OutputConfig, OutputMatrix};

/// An async, shared output routing table.
///
/// Multiple tasks may call `get_assignment` concurrently without blocking each
/// other.  `assign` / `clear_assignment` acquire a write lock for the minimum
/// duration needed to mutate the matrix.
///
/// Clone the `AsyncOutputRouter` to share it across tasks — all clones share
/// the same underlying `Arc<RwLock<…>>`.
#[derive(Clone)]
pub struct AsyncOutputRouter {
    matrix: Arc<RwLock<OutputMatrix>>,
    outputs: Arc<RwLock<Vec<OutputConfig>>>,
}

impl AsyncOutputRouter {
    /// Create a new router with an empty routing matrix.
    pub fn new() -> Self {
        Self {
            matrix: Arc::new(RwLock::new(OutputMatrix::new())),
            outputs: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Add (or replace) an output configuration.
    pub async fn add_output(&self, config: OutputConfig) {
        let mut outputs = self.outputs.write().await;
        if let Some(existing) = outputs.iter_mut().find(|o| o.number == config.number) {
            *existing = config;
        } else {
            outputs.push(config);
        }
    }

    /// Assign `source` to `output_number`.
    ///
    /// Returns the previously assigned source, if any.
    pub async fn assign(&self, output_number: usize, source: usize) -> Option<usize> {
        let mut matrix = self.matrix.write().await;
        matrix.assign(output_number, source)
    }

    /// Clear the assignment for `output_number`.
    ///
    /// Returns the removed source, if any.
    pub async fn clear_assignment(&self, output_number: usize) -> Option<usize> {
        let mut matrix = self.matrix.write().await;
        matrix.clear_assignment(output_number)
    }

    /// Return the source currently assigned to `output_number`.
    pub async fn get_assignment(&self, output_number: usize) -> Option<usize> {
        let matrix = self.matrix.read().await;
        matrix.get_assignment(output_number)
    }

    /// Return all output numbers routing the given `source`.
    pub async fn outputs_for_source(&self, source: usize) -> Vec<usize> {
        let matrix = self.matrix.read().await;
        matrix.outputs_for_source(source)
    }

    /// Apply a passthrough mapping: output `n` → source `n`.
    pub async fn apply_passthrough(&self, output_count: usize) {
        let mut matrix = self.matrix.write().await;
        matrix.apply_passthrough(output_count);
    }

    /// Return the total number of active routing assignments.
    pub async fn assignment_count(&self) -> usize {
        let matrix = self.matrix.read().await;
        matrix.assignment_count()
    }

    /// Clear all routing assignments.
    pub async fn clear_all(&self) {
        let mut matrix = self.matrix.write().await;
        matrix.clear_all();
    }

    /// Route the multiviewer output to the given source.
    ///
    /// Convenience wrapper that sets up output 1 (multiviewer by convention)
    /// as a non-blocking operation suitable for real-time frame loops.
    pub async fn route_multiviewer(&self, source: usize) {
        self.assign(1, source).await;
    }

    /// Return a snapshot of the current routing matrix as a `Vec<(output, source)>` pairs.
    pub async fn snapshot(&self) -> Vec<(usize, usize)> {
        let matrix = self.matrix.read().await;
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        // Collect all assignments by querying for outputs 1..=64 (practical maximum).
        for output in 1..=64 {
            if let Some(source) = matrix.get_assignment(output) {
                pairs.push((output, source));
            }
        }
        pairs
    }

    /// Get the output config for a given output number.
    pub async fn get_output_config(&self, output_number: usize) -> Option<OutputConfig> {
        let outputs = self.outputs.read().await;
        outputs.iter().find(|o| o.number == output_number).cloned()
    }

    /// Deactivate an output (non-destructive — routing is preserved).
    pub async fn deactivate_output(&self, output_number: usize) {
        let mut outputs = self.outputs.write().await;
        if let Some(cfg) = outputs.iter_mut().find(|o| o.number == output_number) {
            cfg.deactivate();
        }
    }
}

impl Default for AsyncOutputRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for AsyncOutputRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncOutputRouter").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output_routing::OutputType;

    #[tokio::test]
    async fn test_assign_and_get() {
        let router = AsyncOutputRouter::new();
        router.assign(1, 5).await;
        assert_eq!(router.get_assignment(1).await, Some(5));
    }

    #[tokio::test]
    async fn test_assignment_count() {
        let router = AsyncOutputRouter::new();
        router.assign(1, 1).await;
        router.assign(2, 3).await;
        assert_eq!(router.assignment_count().await, 2);
    }

    #[tokio::test]
    async fn test_clear_assignment() {
        let router = AsyncOutputRouter::new();
        router.assign(1, 5).await;
        let removed = router.clear_assignment(1).await;
        assert_eq!(removed, Some(5));
        assert_eq!(router.get_assignment(1).await, None);
    }

    #[tokio::test]
    async fn test_passthrough() {
        let router = AsyncOutputRouter::new();
        router.apply_passthrough(4).await;
        assert_eq!(router.assignment_count().await, 4);
        for n in 1..=4 {
            assert_eq!(router.get_assignment(n).await, Some(n));
        }
    }

    #[tokio::test]
    async fn test_outputs_for_source() {
        let router = AsyncOutputRouter::new();
        router.assign(1, 3).await;
        router.assign(2, 3).await;
        router.assign(3, 5).await;
        let mut outs = router.outputs_for_source(3).await;
        outs.sort();
        assert_eq!(outs, vec![1, 2]);
    }

    #[tokio::test]
    async fn test_route_multiviewer() {
        let router = AsyncOutputRouter::new();
        router.route_multiviewer(7).await;
        assert_eq!(router.get_assignment(1).await, Some(7));
    }

    #[tokio::test]
    async fn test_snapshot() {
        let router = AsyncOutputRouter::new();
        router.assign(2, 10).await;
        router.assign(5, 20).await;
        let mut snap = router.snapshot().await;
        snap.sort();
        assert!(snap.contains(&(2, 10)));
        assert!(snap.contains(&(5, 20)));
    }

    #[tokio::test]
    async fn test_add_and_get_output_config() {
        let router = AsyncOutputRouter::new();
        let cfg = OutputConfig::new(1, "PGM", OutputType::Program);
        router.add_output(cfg).await;
        let got = router.get_output_config(1).await;
        assert!(got.is_some());
        assert_eq!(got.as_ref().map(|c| c.number), Some(1));
    }

    #[tokio::test]
    async fn test_deactivate_output() {
        let router = AsyncOutputRouter::new();
        let cfg = OutputConfig::new(1, "PGM", OutputType::Program);
        router.add_output(cfg).await;
        router.deactivate_output(1).await;
        let got = router
            .get_output_config(1)
            .await
            .expect("config should exist");
        assert!(!got.active);
    }

    #[tokio::test]
    async fn test_clear_all() {
        let router = AsyncOutputRouter::new();
        router.apply_passthrough(3).await;
        router.clear_all().await;
        assert_eq!(router.assignment_count().await, 0);
    }

    #[tokio::test]
    async fn test_concurrent_reads_do_not_block() {
        use std::sync::Arc;
        let router = Arc::new(AsyncOutputRouter::new());
        router.assign(1, 42).await;

        // Spawn 8 concurrent read tasks — they should all succeed.
        let mut handles = Vec::new();
        for _ in 0..8 {
            let r = Arc::clone(&router);
            handles.push(tokio::spawn(async move { r.get_assignment(1).await }));
        }
        for h in handles {
            let result = h.await.expect("task should not panic");
            assert_eq!(result, Some(42));
        }
    }

    #[tokio::test]
    async fn test_clone_shares_state() {
        let router = AsyncOutputRouter::new();
        let clone = router.clone();

        router.assign(3, 99).await;
        assert_eq!(clone.get_assignment(3).await, Some(99));
    }
}
