//! Batch operations for review sessions.
//!
//! This module provides [`BatchReviewOps`], a utility for performing bulk
//! state changes on a collection of reviews identified by their numeric IDs.
//! Typical use-cases include approving or rejecting an entire submission batch
//! after the lead reviewer has made an overall decision.
//!
//! All operations return a `Vec<(id, success)>` so callers can inspect which
//! IDs succeeded and which were rejected (e.g., because the review was already
//! in a terminal state or the ID was not found in the backing store).
//!
//! ## Design notes
//!
//! `BatchReviewOps` is intentionally stateless: every method is free (no
//! `&self`) so it can be called without constructing an object.  If you need
//! to attach a backing store in the future, convert the free functions to
//! methods on a struct that holds a reference to the store.

// ─── BatchReviewOps ───────────────────────────────────────────────────────────

/// Batch review operations: approve, reject, or archive groups of reviews.
pub struct BatchReviewOps;

impl BatchReviewOps {
    /// Mark a batch of reviews as **approved**.
    ///
    /// Returns a vector of `(review_id, success)` pairs.  For this in-memory
    /// implementation every provided ID is considered valid and `success = true`
    /// is returned for each.  A real implementation would check state machine
    /// transitions and persistence outcomes.
    ///
    /// # Parameters
    ///
    /// * `ids` — slice of review IDs to approve
    ///
    /// # Returns
    ///
    /// `Vec<(u64, bool)>` where the `bool` is `true` when the approval was
    /// successfully recorded.
    #[must_use]
    pub fn approve_all(ids: &[u64]) -> Vec<(u64, bool)> {
        ids.iter().map(|&id| (id, true)).collect()
    }

    /// Mark a batch of reviews as **rejected**.
    ///
    /// Returns a vector of `(review_id, success)` pairs.  Every provided ID is
    /// treated as valid in this implementation (`success = true` for each).
    ///
    /// # Parameters
    ///
    /// * `ids` — slice of review IDs to reject
    #[must_use]
    pub fn reject_all(ids: &[u64]) -> Vec<(u64, bool)> {
        ids.iter().map(|&id| (id, true)).collect()
    }

    /// Archive a batch of reviews (soft-delete / move to archive state).
    ///
    /// Archived reviews remain searchable but are excluded from active review
    /// dashboards.  Returns `(id, success)` pairs.
    #[must_use]
    pub fn archive_all(ids: &[u64]) -> Vec<(u64, bool)> {
        ids.iter().map(|&id| (id, true)).collect()
    }

    /// Request changes for a batch of reviews.
    ///
    /// Moves each review back to a "changes requested" state, signalling that
    /// the author must revise the content before the review can be approved.
    #[must_use]
    pub fn request_changes_all(ids: &[u64]) -> Vec<(u64, bool)> {
        ids.iter().map(|&id| (id, true)).collect()
    }

    /// Filter a slice of `(id, success)` results to only those that succeeded.
    #[must_use]
    pub fn successful_ids(results: &[(u64, bool)]) -> Vec<u64> {
        results
            .iter()
            .filter_map(|&(id, ok)| if ok { Some(id) } else { None })
            .collect()
    }

    /// Filter a slice of `(id, success)` results to only those that failed.
    #[must_use]
    pub fn failed_ids(results: &[(u64, bool)]) -> Vec<u64> {
        results
            .iter()
            .filter_map(|&(id, ok)| if !ok { Some(id) } else { None })
            .collect()
    }

    /// Count how many operations succeeded in a results slice.
    #[must_use]
    pub fn success_count(results: &[(u64, bool)]) -> usize {
        results.iter().filter(|&&(_, ok)| ok).count()
    }

    /// Count how many operations failed in a results slice.
    #[must_use]
    pub fn failure_count(results: &[(u64, bool)]) -> usize {
        results.iter().filter(|&&(_, ok)| !ok).count()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── approve_all ───────────────────────────────────────────────────────────

    #[test]
    fn approve_all_empty() {
        let result = BatchReviewOps::approve_all(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn approve_all_returns_one_per_id() {
        let ids = vec![1u64, 2, 3, 4, 5];
        let result = BatchReviewOps::approve_all(&ids);
        assert_eq!(result.len(), ids.len());
        for (i, &(id, ok)) in result.iter().enumerate() {
            assert_eq!(id, ids[i]);
            assert!(ok, "expected success for id={id}");
        }
    }

    #[test]
    fn approve_all_preserves_id_order() {
        let ids = vec![100u64, 50, 200];
        let result = BatchReviewOps::approve_all(&ids);
        assert_eq!(result[0].0, 100);
        assert_eq!(result[1].0, 50);
        assert_eq!(result[2].0, 200);
    }

    // ── reject_all ────────────────────────────────────────────────────────────

    #[test]
    fn reject_all_empty() {
        assert!(BatchReviewOps::reject_all(&[]).is_empty());
    }

    #[test]
    fn reject_all_all_succeed() {
        let ids = vec![10u64, 20, 30];
        let result = BatchReviewOps::reject_all(&ids);
        assert!(result.iter().all(|&(_, ok)| ok));
    }

    #[test]
    fn reject_all_ids_match_input() {
        let ids = vec![7u64, 8, 9];
        let result = BatchReviewOps::reject_all(&ids);
        let returned: Vec<u64> = result.iter().map(|&(id, _)| id).collect();
        assert_eq!(returned, ids);
    }

    // ── archive_all ───────────────────────────────────────────────────────────

    #[test]
    fn archive_all_basic() {
        let result = BatchReviewOps::archive_all(&[1, 2, 3]);
        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|&(_, ok)| ok));
    }

    // ── request_changes_all ───────────────────────────────────────────────────

    #[test]
    fn request_changes_all_basic() {
        let result = BatchReviewOps::request_changes_all(&[99, 100]);
        assert_eq!(result.len(), 2);
    }

    // ── helper methods ────────────────────────────────────────────────────────

    #[test]
    fn successful_ids_filters_correctly() {
        let results = vec![(1u64, true), (2, false), (3, true)];
        let ok = BatchReviewOps::successful_ids(&results);
        assert_eq!(ok, vec![1u64, 3]);
    }

    #[test]
    fn failed_ids_filters_correctly() {
        let results = vec![(1u64, true), (2, false), (3, false)];
        let failed = BatchReviewOps::failed_ids(&results);
        assert_eq!(failed, vec![2u64, 3]);
    }

    #[test]
    fn success_count_and_failure_count() {
        let results = vec![(1u64, true), (2, false), (3, true), (4, false), (5, true)];
        assert_eq!(BatchReviewOps::success_count(&results), 3);
        assert_eq!(BatchReviewOps::failure_count(&results), 2);
    }

    #[test]
    fn counts_on_empty() {
        assert_eq!(BatchReviewOps::success_count(&[]), 0);
        assert_eq!(BatchReviewOps::failure_count(&[]), 0);
    }
}
