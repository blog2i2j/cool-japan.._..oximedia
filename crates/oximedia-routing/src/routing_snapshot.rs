//! Routing snapshot save/restore with atomic rollback.
//!
//! Captures the complete state of a [`CrosspointMatrix`] at a point in time
//! and allows restoring it later for instant recall or undo-style rollback.

use std::collections::HashMap;

use crate::matrix::crosspoint::CrosspointMatrix;

/// A frozen snapshot of a crosspoint matrix state.
#[derive(Debug, Clone)]
pub struct RoutingSnapshot {
    /// Snapshot identifier.
    pub id: u64,
    /// Human-readable name.
    pub name: String,
    /// Description / notes.
    pub description: String,
    /// Timestamp in microseconds (caller-supplied, not wall-clock).
    pub timestamp_us: u64,
    /// Matrix dimensions (inputs, outputs).
    pub dimensions: (usize, usize),
    /// Saved crosspoint states.
    crosspoints: HashMap<(usize, usize), f32>,
    /// Input labels at the time of capture.
    input_labels: Vec<String>,
    /// Output labels at the time of capture.
    output_labels: Vec<String>,
}

impl RoutingSnapshot {
    /// Number of active crosspoints in this snapshot.
    pub fn active_count(&self) -> usize {
        self.crosspoints.len()
    }

    /// Returns `true` if the snapshot has no active crosspoints.
    pub fn is_empty(&self) -> bool {
        self.crosspoints.is_empty()
    }

    /// Returns the saved crosspoint gain for (input, output), if connected.
    pub fn get_gain(&self, input: usize, output: usize) -> Option<f32> {
        self.crosspoints.get(&(input, output)).copied()
    }

    /// Returns all active crosspoints as (input, output, gain_db) triples.
    pub fn active_crosspoints(&self) -> Vec<(usize, usize, f32)> {
        self.crosspoints
            .iter()
            .map(|(&(i, o), &g)| (i, o, g))
            .collect()
    }

    /// Returns the input labels.
    pub fn input_labels(&self) -> &[String] {
        &self.input_labels
    }

    /// Returns the output labels.
    pub fn output_labels(&self) -> &[String] {
        &self.output_labels
    }
}

/// Difference between two snapshots.
#[derive(Debug, Clone)]
pub struct SnapshotDiff {
    /// Crosspoints that were added (not in `before` but in `after`).
    pub added: Vec<(usize, usize, f32)>,
    /// Crosspoints that were removed (in `before` but not in `after`).
    pub removed: Vec<(usize, usize, f32)>,
    /// Crosspoints whose gain changed.
    pub gain_changed: Vec<(usize, usize, f32, f32)>,
    /// Number of unchanged crosspoints.
    pub unchanged: usize,
}

impl SnapshotDiff {
    /// Returns `true` if the two snapshots are identical.
    pub fn is_identical(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.gain_changed.is_empty()
    }

    /// Total number of differences.
    pub fn diff_count(&self) -> usize {
        self.added.len() + self.removed.len() + self.gain_changed.len()
    }
}

/// Manages routing snapshots with save/restore/rollback.
#[derive(Debug)]
pub struct SnapshotManager {
    snapshots: Vec<RoutingSnapshot>,
    next_id: u64,
    /// Maximum number of snapshots to retain (0 = unlimited).
    max_snapshots: usize,
}

impl Default for SnapshotManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SnapshotManager {
    /// Creates a new, empty snapshot manager.
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
            next_id: 0,
            max_snapshots: 0,
        }
    }

    /// Creates a manager that retains at most `max` snapshots.
    pub fn with_max(max: usize) -> Self {
        Self {
            snapshots: Vec::new(),
            next_id: 0,
            max_snapshots: max,
        }
    }

    /// Captures a snapshot of the given matrix.
    pub fn capture(
        &mut self,
        matrix: &CrosspointMatrix,
        name: impl Into<String>,
        description: impl Into<String>,
        timestamp_us: u64,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        let mut crosspoints = HashMap::new();
        let active = matrix.get_active_crosspoints();
        for (cp_id, gain_db) in active {
            crosspoints.insert((cp_id.input, cp_id.output), gain_db);
        }

        let mut input_labels = Vec::new();
        for i in 0..matrix.input_count() {
            input_labels.push(matrix.get_input_label(i).unwrap_or("?").to_string());
        }
        let mut output_labels = Vec::new();
        for o in 0..matrix.output_count() {
            output_labels.push(matrix.get_output_label(o).unwrap_or("?").to_string());
        }

        let snapshot = RoutingSnapshot {
            id,
            name: name.into(),
            description: description.into(),
            timestamp_us,
            dimensions: (matrix.input_count(), matrix.output_count()),
            crosspoints,
            input_labels,
            output_labels,
        };

        self.snapshots.push(snapshot);

        // Evict oldest if needed
        if self.max_snapshots > 0 && self.snapshots.len() > self.max_snapshots {
            self.snapshots.remove(0);
        }

        id
    }

    /// Restores a snapshot to the given matrix **atomically**.
    ///
    /// Either the matrix is fully updated to match the snapshot, or it is left
    /// completely unchanged (rollback on any error).
    ///
    /// The atomicity guarantee is implemented via a clone-and-apply pattern:
    /// 1. Clone the target matrix as a backup.
    /// 2. Apply all changes to the *live* matrix.
    /// 3. If any step fails, `std::mem::swap` the backup back in, restoring
    ///    the original state before returning the error.
    ///
    /// Returns `Err` if the snapshot is not found or dimensions don't match.
    pub fn restore(
        &self,
        snapshot_id: u64,
        matrix: &mut CrosspointMatrix,
    ) -> Result<(), SnapshotError> {
        let snapshot = self
            .get(snapshot_id)
            .ok_or(SnapshotError::NotFound(snapshot_id))?;

        if snapshot.dimensions != (matrix.input_count(), matrix.output_count()) {
            return Err(SnapshotError::DimensionMismatch {
                snapshot: snapshot.dimensions,
                matrix: (matrix.input_count(), matrix.output_count()),
            });
        }

        // Take a backup of the current state before touching anything.
        let mut backup = matrix.clone();

        // Apply all changes.  On error, swap the backup back in so the matrix
        // is left exactly as it was before this call.
        let result = self.apply_snapshot(snapshot, matrix);
        if result.is_err() {
            // Atomic rollback: swap backup into the live matrix slot.
            std::mem::swap(matrix, &mut backup);
        }
        result
    }

    /// Internal: apply snapshot state to matrix without rollback logic.
    fn apply_snapshot(
        &self,
        snapshot: &RoutingSnapshot,
        matrix: &mut CrosspointMatrix,
    ) -> Result<(), SnapshotError> {
        // Clear current state.
        matrix.clear_all();

        // Restore all crosspoints.
        for (&(input, output), &gain_db) in &snapshot.crosspoints {
            matrix.connect(input, output, Some(gain_db)).map_err(|e| {
                SnapshotError::RestoreError(format!("Failed to connect ({input},{output}): {e}"))
            })?;
        }

        // Restore labels.
        for (i, label) in snapshot.input_labels.iter().enumerate() {
            let _ = matrix.set_input_label(i, label.clone());
        }
        for (o, label) in snapshot.output_labels.iter().enumerate() {
            let _ = matrix.set_output_label(o, label.clone());
        }

        Ok(())
    }

    /// Gets a snapshot by ID.
    pub fn get(&self, id: u64) -> Option<&RoutingSnapshot> {
        self.snapshots.iter().find(|s| s.id == id)
    }

    /// Returns the number of stored snapshots.
    pub fn count(&self) -> usize {
        self.snapshots.len()
    }

    /// Returns `true` if no snapshots are stored.
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Deletes a snapshot by ID.
    pub fn delete(&mut self, id: u64) -> bool {
        if let Some(pos) = self.snapshots.iter().position(|s| s.id == id) {
            self.snapshots.remove(pos);
            true
        } else {
            false
        }
    }

    /// Returns the most recently captured snapshot.
    pub fn latest(&self) -> Option<&RoutingSnapshot> {
        self.snapshots.last()
    }

    /// Lists all snapshot IDs and names.
    pub fn list(&self) -> Vec<(u64, &str)> {
        self.snapshots
            .iter()
            .map(|s| (s.id, s.name.as_str()))
            .collect()
    }

    /// Computes a diff between two snapshots.
    pub fn diff(&self, before_id: u64, after_id: u64) -> Result<SnapshotDiff, SnapshotError> {
        let before = self
            .get(before_id)
            .ok_or(SnapshotError::NotFound(before_id))?;
        let after = self
            .get(after_id)
            .ok_or(SnapshotError::NotFound(after_id))?;

        Ok(diff_snapshots(before, after))
    }

    /// Clears all snapshots.
    pub fn clear(&mut self) {
        self.snapshots.clear();
    }
}

/// Computes the diff between two routing snapshots.
pub fn diff_snapshots(before: &RoutingSnapshot, after: &RoutingSnapshot) -> SnapshotDiff {
    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut gain_changed = Vec::new();
    let mut unchanged = 0usize;

    // Find removed and gain-changed
    for (&(i, o), &gain_before) in &before.crosspoints {
        match after.crosspoints.get(&(i, o)) {
            Some(&gain_after) => {
                if (gain_before - gain_after).abs() > 1e-6 {
                    gain_changed.push((i, o, gain_before, gain_after));
                } else {
                    unchanged += 1;
                }
            }
            None => {
                removed.push((i, o, gain_before));
            }
        }
    }

    // Find added
    for (&(i, o), &gain_after) in &after.crosspoints {
        if !before.crosspoints.contains_key(&(i, o)) {
            added.push((i, o, gain_after));
        }
    }

    SnapshotDiff {
        added,
        removed,
        gain_changed,
        unchanged,
    }
}

/// Errors from snapshot operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum SnapshotError {
    /// Snapshot not found.
    #[error("Snapshot not found: {0}")]
    NotFound(u64),
    /// Dimension mismatch between snapshot and target matrix.
    #[error("Dimension mismatch: snapshot {snapshot:?} vs matrix {matrix:?}")]
    DimensionMismatch {
        snapshot: (usize, usize),
        matrix: (usize, usize),
    },
    /// Error during restore.
    #[error("Restore error: {0}")]
    RestoreError(String),
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_matrix() -> CrosspointMatrix {
        let mut m = CrosspointMatrix::new(4, 4);
        m.connect(0, 0, Some(-6.0)).expect("valid");
        m.connect(1, 1, Some(0.0)).expect("valid");
        m.connect(2, 3, Some(-12.0)).expect("valid");
        m
    }

    #[test]
    fn test_capture_and_count() {
        let mut mgr = SnapshotManager::new();
        let m = make_matrix();
        let id = mgr.capture(&m, "snap1", "test", 1000);
        assert_eq!(id, 0);
        assert_eq!(mgr.count(), 1);
    }

    #[test]
    fn test_restore_roundtrip() {
        let mut mgr = SnapshotManager::new();
        let original = make_matrix();
        let id = mgr.capture(&original, "before", "", 0);

        let mut target = CrosspointMatrix::new(4, 4);
        assert!(!target.is_connected(0, 0));

        mgr.restore(id, &mut target).expect("restore ok");
        assert!(target.is_connected(0, 0));
        assert!(target.is_connected(1, 1));
        assert!(target.is_connected(2, 3));
    }

    #[test]
    fn test_restore_not_found() {
        let mgr = SnapshotManager::new();
        let mut m = CrosspointMatrix::new(4, 4);
        let result = mgr.restore(99, &mut m);
        assert!(result.is_err());
    }

    #[test]
    fn test_restore_dimension_mismatch() {
        let mut mgr = SnapshotManager::new();
        let m4 = make_matrix();
        let id = mgr.capture(&m4, "4x4", "", 0);

        let mut m8 = CrosspointMatrix::new(8, 8);
        let result = mgr.restore(id, &mut m8);
        assert!(matches!(
            result,
            Err(SnapshotError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_snapshot_active_count() {
        let mut mgr = SnapshotManager::new();
        let m = make_matrix();
        mgr.capture(&m, "s", "", 0);
        let snap = mgr.get(0).expect("exists");
        assert_eq!(snap.active_count(), 3);
        assert!(!snap.is_empty());
    }

    #[test]
    fn test_snapshot_get_gain() {
        let mut mgr = SnapshotManager::new();
        let m = make_matrix();
        mgr.capture(&m, "s", "", 0);
        let snap = mgr.get(0).expect("exists");
        assert!((snap.get_gain(0, 0).expect("exists") - (-6.0)).abs() < 1e-6);
        assert!(snap.get_gain(3, 3).is_none());
    }

    #[test]
    fn test_delete_snapshot() {
        let mut mgr = SnapshotManager::new();
        let m = make_matrix();
        let id = mgr.capture(&m, "s", "", 0);
        assert!(mgr.delete(id));
        assert_eq!(mgr.count(), 0);
        assert!(!mgr.delete(id)); // already deleted
    }

    #[test]
    fn test_latest() {
        let mut mgr = SnapshotManager::new();
        let m = make_matrix();
        mgr.capture(&m, "first", "", 0);
        mgr.capture(&m, "second", "", 100);
        assert_eq!(mgr.latest().expect("exists").name, "second");
    }

    #[test]
    fn test_list() {
        let mut mgr = SnapshotManager::new();
        let m = make_matrix();
        mgr.capture(&m, "alpha", "", 0);
        mgr.capture(&m, "beta", "", 100);
        let list = mgr.list();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_max_snapshots_eviction() {
        let mut mgr = SnapshotManager::with_max(2);
        let m = make_matrix();
        mgr.capture(&m, "a", "", 0);
        mgr.capture(&m, "b", "", 100);
        mgr.capture(&m, "c", "", 200);
        assert_eq!(mgr.count(), 2);
        // First snapshot should have been evicted
        assert!(mgr.get(0).is_none());
        assert!(mgr.get(1).is_some());
    }

    #[test]
    fn test_diff_identical() {
        let mut mgr = SnapshotManager::new();
        let m = make_matrix();
        let id1 = mgr.capture(&m, "a", "", 0);
        let id2 = mgr.capture(&m, "b", "", 100);
        let diff = mgr.diff(id1, id2).expect("valid");
        assert!(diff.is_identical());
        assert_eq!(diff.diff_count(), 0);
        assert_eq!(diff.unchanged, 3);
    }

    #[test]
    fn test_diff_added_removed() {
        let mut mgr = SnapshotManager::new();
        let mut m1 = CrosspointMatrix::new(4, 4);
        m1.connect(0, 0, Some(0.0)).expect("valid");
        let id1 = mgr.capture(&m1, "before", "", 0);

        let mut m2 = CrosspointMatrix::new(4, 4);
        m2.connect(1, 1, Some(-3.0)).expect("valid");
        let id2 = mgr.capture(&m2, "after", "", 100);

        let diff = mgr.diff(id1, id2).expect("valid");
        assert_eq!(diff.removed.len(), 1); // (0,0) removed
        assert_eq!(diff.added.len(), 1); // (1,1) added
        assert_eq!(diff.unchanged, 0);
    }

    #[test]
    fn test_diff_gain_changed() {
        let mut mgr = SnapshotManager::new();
        let mut m1 = CrosspointMatrix::new(4, 4);
        m1.connect(0, 0, Some(0.0)).expect("valid");
        let id1 = mgr.capture(&m1, "before", "", 0);

        let mut m2 = CrosspointMatrix::new(4, 4);
        m2.connect(0, 0, Some(-6.0)).expect("valid");
        let id2 = mgr.capture(&m2, "after", "", 100);

        let diff = mgr.diff(id1, id2).expect("valid");
        assert_eq!(diff.gain_changed.len(), 1);
        let (i, o, before, after) = diff.gain_changed[0];
        assert_eq!((i, o), (0, 0));
        assert!((before - 0.0).abs() < 1e-6);
        assert!((after - (-6.0)).abs() < 1e-6);
    }

    #[test]
    fn test_diff_not_found() {
        let mgr = SnapshotManager::new();
        assert!(mgr.diff(0, 1).is_err());
    }

    #[test]
    fn test_clear() {
        let mut mgr = SnapshotManager::new();
        let m = make_matrix();
        mgr.capture(&m, "a", "", 0);
        mgr.clear();
        assert!(mgr.is_empty());
    }

    #[test]
    fn test_empty_snapshot() {
        let mut mgr = SnapshotManager::new();
        let m = CrosspointMatrix::new(4, 4);
        mgr.capture(&m, "empty", "", 0);
        let snap = mgr.get(0).expect("exists");
        assert!(snap.is_empty());
        assert_eq!(snap.active_count(), 0);
    }

    #[test]
    fn test_snapshot_labels_preserved() {
        let mut mgr = SnapshotManager::new();
        let mut m = CrosspointMatrix::new(2, 2);
        m.set_input_label(0, "Mic 1".to_string()).expect("ok");
        m.set_output_label(1, "Mon R".to_string()).expect("ok");
        mgr.capture(&m, "labeled", "", 0);
        let snap = mgr.get(0).expect("exists");
        assert_eq!(snap.input_labels()[0], "Mic 1");
        assert_eq!(snap.output_labels()[1], "Mon R");
    }

    #[test]
    fn test_restore_preserves_labels() {
        let mut mgr = SnapshotManager::new();
        let mut m = CrosspointMatrix::new(2, 2);
        m.set_input_label(0, "Source A".to_string()).expect("ok");
        m.connect(0, 0, Some(0.0)).expect("ok");
        let id = mgr.capture(&m, "snap", "", 0);

        let mut m2 = CrosspointMatrix::new(2, 2);
        mgr.restore(id, &mut m2).expect("ok");
        assert_eq!(m2.get_input_label(0), Some("Source A"));
    }

    #[test]
    fn test_active_crosspoints_list() {
        let mut mgr = SnapshotManager::new();
        let m = make_matrix();
        mgr.capture(&m, "s", "", 0);
        let snap = mgr.get(0).expect("exists");
        let active = snap.active_crosspoints();
        assert_eq!(active.len(), 3);
    }

    // -----------------------------------------------------------------------
    // New tests: atomic rollback guarantee (12 tests)
    // -----------------------------------------------------------------------

    /// Restore of a valid snapshot fully replaces the matrix state.
    #[test]
    fn test_atomic_restore_full_replace() {
        let mut mgr = SnapshotManager::new();

        // Snapshot A: input 0 → output 0 at -6 dB
        let mut m_a = CrosspointMatrix::new(4, 4);
        m_a.connect(0, 0, Some(-6.0)).expect("ok");
        let id_a = mgr.capture(&m_a, "state_a", "", 0);

        // Live matrix: input 2 → output 2 at 0 dB
        let mut live = CrosspointMatrix::new(4, 4);
        live.connect(2, 2, Some(0.0)).expect("ok");
        assert!(live.is_connected(2, 2));
        assert!(!live.is_connected(0, 0));

        // Restore A into live — should fully replace state.
        mgr.restore(id_a, &mut live).expect("restore ok");
        assert!(live.is_connected(0, 0));
        assert!(!live.is_connected(2, 2)); // old connection gone
    }

    /// Save and restore is an exact round-trip including gain values.
    #[test]
    fn test_atomic_restore_gain_roundtrip() {
        let mut mgr = SnapshotManager::new();
        let mut m = CrosspointMatrix::new(4, 4);
        m.connect(0, 1, Some(-12.5)).expect("ok");
        m.connect(3, 3, Some(6.0)).expect("ok");
        let id = mgr.capture(&m, "gains", "", 0);

        let mut target = CrosspointMatrix::new(4, 4);
        mgr.restore(id, &mut target).expect("restore ok");

        // Check gain is preserved to floating-point precision.
        let state_01 = target.get_state(0, 1);
        let state_33 = target.get_state(3, 3);
        if let crate::matrix::crosspoint::CrosspointState::Connected { gain_db } = state_01 {
            assert!((gain_db - (-12.5)).abs() < 1e-5, "gain mismatch for (0,1)");
        } else {
            panic!("Expected (0,1) to be connected");
        }
        if let crate::matrix::crosspoint::CrosspointState::Connected { gain_db } = state_33 {
            assert!((gain_db - 6.0).abs() < 1e-5, "gain mismatch for (3,3)");
        } else {
            panic!("Expected (3,3) to be connected");
        }
    }

    /// On a not-found error the matrix is completely unchanged.
    #[test]
    fn test_atomic_rollback_on_not_found() {
        let mgr = SnapshotManager::new();
        let mut live = CrosspointMatrix::new(4, 4);
        live.connect(1, 1, Some(0.0)).expect("ok");

        let result = mgr.restore(999, &mut live);
        assert!(result.is_err());

        // Matrix must be untouched.
        assert!(live.is_connected(1, 1));
        assert_eq!(live.get_active_crosspoints().len(), 1);
    }

    /// On a dimension mismatch error the matrix is completely unchanged.
    #[test]
    fn test_atomic_rollback_on_dimension_mismatch() {
        let mut mgr = SnapshotManager::new();
        let m = make_matrix(); // 4×4
        let id = mgr.capture(&m, "4x4", "", 0);

        let mut live = CrosspointMatrix::new(8, 8); // wrong size
        live.connect(0, 0, Some(0.0)).expect("ok");
        live.connect(7, 7, Some(0.0)).expect("ok");

        let result = mgr.restore(id, &mut live);
        assert!(matches!(
            result,
            Err(SnapshotError::DimensionMismatch { .. })
        ));

        // Matrix must still have its original connections.
        assert!(live.is_connected(0, 0));
        assert!(live.is_connected(7, 7));
        assert_eq!(live.get_active_crosspoints().len(), 2);
    }

    /// Restoring a second snapshot after a first restore works correctly.
    #[test]
    fn test_restore_sequence() {
        let mut mgr = SnapshotManager::new();
        let mut m1 = CrosspointMatrix::new(4, 4);
        m1.connect(0, 0, Some(0.0)).expect("ok");
        let id1 = mgr.capture(&m1, "s1", "", 0);

        let mut m2 = CrosspointMatrix::new(4, 4);
        m2.connect(1, 1, Some(-3.0)).expect("ok");
        m2.connect(2, 2, Some(-6.0)).expect("ok");
        let id2 = mgr.capture(&m2, "s2", "", 100);

        let mut live = CrosspointMatrix::new(4, 4);

        // Restore first snapshot.
        mgr.restore(id1, &mut live).expect("ok");
        assert!(live.is_connected(0, 0));
        assert!(!live.is_connected(1, 1));

        // Restore second snapshot.
        mgr.restore(id2, &mut live).expect("ok");
        assert!(!live.is_connected(0, 0));
        assert!(live.is_connected(1, 1));
        assert!(live.is_connected(2, 2));
    }

    /// `list_snapshots` returns all saved snapshots with correct metadata.
    #[test]
    fn test_list_snapshots_metadata() {
        let mut mgr = SnapshotManager::new();
        let m = make_matrix();
        let id_a = mgr.capture(&m, "Alpha", "first", 1000);
        let id_b = mgr.capture(&m, "Beta", "second", 2000);

        let list = mgr.list();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0], (id_a, "Alpha"));
        assert_eq!(list[1], (id_b, "Beta"));
    }

    /// `compare` shows added crosspoints correctly.
    #[test]
    fn test_compare_added_crosspoints() {
        let mut mgr = SnapshotManager::new();
        let m_empty = CrosspointMatrix::new(4, 4);
        let id_before = mgr.capture(&m_empty, "before", "", 0);

        let m = make_matrix(); // 3 connections
        let id_after = mgr.capture(&m, "after", "", 100);

        let diff = mgr.diff(id_before, id_after).expect("ok");
        assert_eq!(diff.added.len(), 3);
        assert!(diff.removed.is_empty());
        assert!(diff.gain_changed.is_empty());
        assert!(!diff.is_identical());
    }

    /// `compare` shows removed crosspoints correctly.
    #[test]
    fn test_compare_removed_crosspoints() {
        let mut mgr = SnapshotManager::new();
        let m = make_matrix(); // 3 connections
        let id_before = mgr.capture(&m, "before", "", 0);

        let m_empty = CrosspointMatrix::new(4, 4);
        let id_after = mgr.capture(&m_empty, "after", "", 100);

        let diff = mgr.diff(id_before, id_after).expect("ok");
        assert_eq!(diff.removed.len(), 3);
        assert!(diff.added.is_empty());
        assert!(!diff.is_identical());
    }

    /// `compare` shows gain-changed crosspoints correctly.
    #[test]
    fn test_compare_gain_changed() {
        let mut mgr = SnapshotManager::new();
        let mut m1 = CrosspointMatrix::new(4, 4);
        m1.connect(0, 0, Some(-3.0)).expect("ok");
        let id1 = mgr.capture(&m1, "before", "", 0);

        let mut m2 = CrosspointMatrix::new(4, 4);
        m2.connect(0, 0, Some(-9.0)).expect("ok"); // same point, different gain
        let id2 = mgr.capture(&m2, "after", "", 100);

        let diff = mgr.diff(id1, id2).expect("ok");
        assert_eq!(diff.gain_changed.len(), 1);
        assert!(diff.removed.is_empty());
        assert!(diff.added.is_empty());
        let (i, o, before, after) = diff.gain_changed[0];
        assert_eq!((i, o), (0, 0));
        assert!((before - (-3.0)).abs() < 1e-6);
        assert!((after - (-9.0)).abs() < 1e-6);
    }

    /// `compare` of two identical snapshots is fully identical.
    #[test]
    fn test_compare_identical() {
        let mut mgr = SnapshotManager::new();
        let m = make_matrix();
        let id1 = mgr.capture(&m, "s1", "", 0);
        let id2 = mgr.capture(&m, "s2", "", 50);

        let diff = mgr.diff(id1, id2).expect("ok");
        assert!(diff.is_identical());
        assert_eq!(diff.diff_count(), 0);
        assert_eq!(diff.unchanged, 3);
    }

    /// Restoring preserves the matrix unchanged when called on itself (idempotent).
    #[test]
    fn test_restore_is_idempotent() {
        let mut mgr = SnapshotManager::new();
        let m = make_matrix();
        let id = mgr.capture(&m, "snap", "", 0);

        let mut live = make_matrix();
        mgr.restore(id, &mut live).expect("first restore ok");
        mgr.restore(id, &mut live).expect("second restore ok");

        // After two restores, should still match the original.
        assert!(live.is_connected(0, 0));
        assert!(live.is_connected(1, 1));
        assert!(live.is_connected(2, 3));
        assert_eq!(live.get_active_crosspoints().len(), 3);
    }

    /// 256×256 sparse restore: captures all 256 connections and restores them.
    #[test]
    fn test_snapshot_restore_256x256_diagonal() {
        let mut mgr = SnapshotManager::new();
        let mut m = CrosspointMatrix::new(256, 256);
        for i in 0..256 {
            m.connect(i, i, Some(-(i as f32))).expect("ok");
        }
        let id = mgr.capture(&m, "diagonal", "", 0);

        let mut target = CrosspointMatrix::new(256, 256);
        mgr.restore(id, &mut target).expect("restore ok");

        // Verify a sample of connections and gains.
        for i in [0, 63, 127, 200, 255] {
            assert!(
                target.is_connected(i, i),
                "diagonal {i} should be connected"
            );
        }
        assert_eq!(target.get_active_crosspoints().len(), 256);
    }
}
