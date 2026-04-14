//! Review deadline tracking and enforcement.
//!
//! This module provides [`ReviewDeadline`], a lightweight struct for attaching
//! a due date to a review and computing whether the deadline has passed.
//!
//! Timestamps throughout this module are expressed as **Unix epoch seconds**
//! (seconds since 1970-01-01T00:00:00 UTC) so that they remain compatible with
//! any external clock source without pulling in heavy time-zone dependencies.
//!
//! ## Example
//!
//! ```rust
//! use oximedia_review::deadline::ReviewDeadline;
//!
//! let now_ts = 1_700_000_000u64; // some Unix timestamp
//! let due_ts = now_ts + 3600;    // deadline one hour from now
//! let deadline = ReviewDeadline::new(42, due_ts);
//!
//! assert!(!deadline.is_overdue(now_ts));
//! assert!(deadline.time_remaining_s(now_ts) > 0);
//! assert!(deadline.is_overdue(due_ts + 1));
//! ```

// ─── ReviewDeadline ───────────────────────────────────────────────────────────

/// A deadline attached to a specific review.
///
/// Deadlines are identified by the review they belong to (`review_id`) and a
/// Unix epoch timestamp (`due_ts`) marking the point after which the review is
/// considered overdue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReviewDeadline {
    /// Identifier of the review this deadline belongs to.
    pub review_id: u64,
    /// Unix epoch timestamp (seconds) when the review is due.
    pub due_ts: u64,
    /// Optional human-readable label for the deadline phase
    /// (e.g. "internal review", "client approval").
    pub label: Option<String>,
}

impl ReviewDeadline {
    /// Create a new deadline for the given review ID with the specified due
    /// Unix epoch timestamp.
    #[must_use]
    pub fn new(review_id: u64, due_ts: u64) -> Self {
        Self {
            review_id,
            due_ts,
            label: None,
        }
    }

    /// Attach a human-readable label to this deadline (builder pattern).
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Returns `true` if `now_ts` is strictly after `due_ts`.
    ///
    /// # Parameters
    ///
    /// * `now_ts` — current time as a Unix epoch timestamp in seconds
    #[must_use]
    pub fn is_overdue(&self, now_ts: u64) -> bool {
        now_ts > self.due_ts
    }

    /// Compute the number of seconds remaining until the deadline.
    ///
    /// * Positive return value: seconds until the deadline.
    /// * Zero: deadline is exactly now.
    /// * Negative return value: seconds past the deadline (i.e. overdue by that many seconds).
    ///
    /// # Parameters
    ///
    /// * `now_ts` — current time as a Unix epoch timestamp in seconds
    #[must_use]
    pub fn time_remaining_s(&self, now_ts: u64) -> i64 {
        self.due_ts as i64 - now_ts as i64
    }

    /// Human-friendly description of the time remaining.
    ///
    /// Returns strings such as `"2d 03h 15m"`, `"45m 10s"`, `"overdue by 1h 30m"`.
    #[must_use]
    pub fn time_remaining_human(&self, now_ts: u64) -> String {
        let remaining = self.time_remaining_s(now_ts);
        if remaining < 0 {
            let overdue = (-remaining) as u64;
            format!("overdue by {}", format_seconds(overdue))
        } else {
            format_seconds(remaining as u64)
        }
    }

    /// Returns `true` if the deadline falls within the next `warn_before_s` seconds.
    ///
    /// Useful for generating "deadline approaching" notifications.
    #[must_use]
    pub fn is_approaching(&self, now_ts: u64, warn_before_s: u64) -> bool {
        let remaining = self.time_remaining_s(now_ts);
        remaining >= 0 && (remaining as u64) <= warn_before_s
    }
}

// ─── Deadline collection ──────────────────────────────────────────────────────

/// A collection of deadlines with batch query support.
#[derive(Debug, Default, Clone)]
pub struct DeadlineCollection {
    deadlines: Vec<ReviewDeadline>,
}

impl DeadlineCollection {
    /// Create an empty collection.
    #[must_use]
    pub fn new() -> Self {
        Self {
            deadlines: Vec::new(),
        }
    }

    /// Add a deadline to the collection.
    pub fn add(&mut self, deadline: ReviewDeadline) {
        self.deadlines.push(deadline);
    }

    /// Remove all deadlines for the given review ID.
    pub fn remove_for_review(&mut self, review_id: u64) {
        self.deadlines.retain(|d| d.review_id != review_id);
    }

    /// Return all deadlines that are overdue at `now_ts`.
    #[must_use]
    pub fn overdue_at(&self, now_ts: u64) -> Vec<&ReviewDeadline> {
        self.deadlines
            .iter()
            .filter(|d| d.is_overdue(now_ts))
            .collect()
    }

    /// Return all deadlines approaching within `warn_before_s` of `now_ts`.
    #[must_use]
    pub fn approaching_at(&self, now_ts: u64, warn_before_s: u64) -> Vec<&ReviewDeadline> {
        self.deadlines
            .iter()
            .filter(|d| d.is_approaching(now_ts, warn_before_s))
            .collect()
    }

    /// Return the deadline for a specific review, if one exists.
    #[must_use]
    pub fn get(&self, review_id: u64) -> Option<&ReviewDeadline> {
        self.deadlines.iter().find(|d| d.review_id == review_id)
    }

    /// Total number of deadlines in the collection.
    #[must_use]
    pub fn len(&self) -> usize {
        self.deadlines.len()
    }

    /// Returns `true` if the collection contains no deadlines.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.deadlines.is_empty()
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Format a duration in seconds as a human-readable string `Xd Xh Xm Xs`.
fn format_seconds(total_s: u64) -> String {
    if total_s == 0 {
        return "0s".to_string();
    }
    let days = total_s / 86400;
    let hours = (total_s % 86400) / 3600;
    let minutes = (total_s % 3600) / 60;
    let seconds = total_s % 60;

    let mut parts = Vec::new();
    if days > 0 {
        parts.push(format!("{days}d"));
    }
    if hours > 0 {
        parts.push(format!("{hours:02}h"));
    }
    if minutes > 0 {
        parts.push(format!("{minutes:02}m"));
    }
    if seconds > 0 || parts.is_empty() {
        parts.push(format!("{seconds:02}s"));
    }
    parts.join(" ")
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const BASE_TS: u64 = 1_700_000_000;

    // ── ReviewDeadline::new ───────────────────────────────────────────────────

    #[test]
    fn new_stores_fields() {
        let d = ReviewDeadline::new(42, BASE_TS + 3600);
        assert_eq!(d.review_id, 42);
        assert_eq!(d.due_ts, BASE_TS + 3600);
        assert!(d.label.is_none());
    }

    #[test]
    fn with_label_sets_label() {
        let d = ReviewDeadline::new(1, BASE_TS).with_label("Internal review");
        assert_eq!(d.label.as_deref(), Some("Internal review"));
    }

    // ── is_overdue ────────────────────────────────────────────────────────────

    #[test]
    fn not_overdue_before_deadline() {
        let d = ReviewDeadline::new(1, BASE_TS + 100);
        assert!(!d.is_overdue(BASE_TS));
        assert!(!d.is_overdue(BASE_TS + 99));
    }

    #[test]
    fn not_overdue_at_exact_deadline() {
        let d = ReviewDeadline::new(1, BASE_TS);
        assert!(!d.is_overdue(BASE_TS)); // strictly after
    }

    #[test]
    fn overdue_after_deadline() {
        let d = ReviewDeadline::new(1, BASE_TS);
        assert!(d.is_overdue(BASE_TS + 1));
        assert!(d.is_overdue(BASE_TS + 999_999));
    }

    // ── time_remaining_s ──────────────────────────────────────────────────────

    #[test]
    fn time_remaining_positive_when_not_overdue() {
        let d = ReviewDeadline::new(1, BASE_TS + 3600);
        assert_eq!(d.time_remaining_s(BASE_TS), 3600);
    }

    #[test]
    fn time_remaining_zero_at_exact_deadline() {
        let d = ReviewDeadline::new(1, BASE_TS);
        assert_eq!(d.time_remaining_s(BASE_TS), 0);
    }

    #[test]
    fn time_remaining_negative_when_overdue() {
        let d = ReviewDeadline::new(1, BASE_TS);
        assert_eq!(d.time_remaining_s(BASE_TS + 60), -60);
    }

    // ── time_remaining_human ──────────────────────────────────────────────────

    #[test]
    fn human_format_future() {
        let d = ReviewDeadline::new(1, BASE_TS + 90061); // 1d 1h 1m 1s
        let s = d.time_remaining_human(BASE_TS);
        assert!(s.contains('d'), "expected days in '{s}'");
        assert!(s.contains('h'), "expected hours in '{s}'");
    }

    #[test]
    fn human_format_overdue() {
        let d = ReviewDeadline::new(1, BASE_TS);
        let s = d.time_remaining_human(BASE_TS + 3600);
        assert!(s.starts_with("overdue by"), "got '{s}'");
    }

    // ── is_approaching ────────────────────────────────────────────────────────

    #[test]
    fn approaching_within_window() {
        let d = ReviewDeadline::new(1, BASE_TS + 300);
        assert!(d.is_approaching(BASE_TS, 3600)); // 300s remaining, warn at 3600
        assert!(!d.is_approaching(BASE_TS, 100)); // 300s remaining, warn at 100
    }

    #[test]
    fn not_approaching_when_overdue() {
        let d = ReviewDeadline::new(1, BASE_TS);
        assert!(!d.is_approaching(BASE_TS + 1, 3600)); // already overdue
    }

    // ── DeadlineCollection ────────────────────────────────────────────────────

    #[test]
    fn collection_add_and_len() {
        let mut col = DeadlineCollection::new();
        assert!(col.is_empty());
        col.add(ReviewDeadline::new(1, BASE_TS + 1000));
        col.add(ReviewDeadline::new(2, BASE_TS + 2000));
        assert_eq!(col.len(), 2);
        assert!(!col.is_empty());
    }

    #[test]
    fn collection_get_by_review_id() {
        let mut col = DeadlineCollection::new();
        col.add(ReviewDeadline::new(10, BASE_TS + 500));
        col.add(ReviewDeadline::new(20, BASE_TS + 1000));
        assert_eq!(col.get(10).map(|d| d.review_id), Some(10));
        assert!(col.get(99).is_none());
    }

    #[test]
    fn collection_remove_for_review() {
        let mut col = DeadlineCollection::new();
        col.add(ReviewDeadline::new(1, BASE_TS + 100));
        col.add(ReviewDeadline::new(2, BASE_TS + 200));
        col.remove_for_review(1);
        assert_eq!(col.len(), 1);
        assert!(col.get(1).is_none());
    }

    #[test]
    fn collection_overdue_at() {
        let mut col = DeadlineCollection::new();
        col.add(ReviewDeadline::new(1, BASE_TS - 100)); // already overdue
        col.add(ReviewDeadline::new(2, BASE_TS + 100)); // not yet due
        let overdue = col.overdue_at(BASE_TS);
        assert_eq!(overdue.len(), 1);
        assert_eq!(overdue[0].review_id, 1);
    }

    #[test]
    fn collection_approaching_at() {
        let mut col = DeadlineCollection::new();
        col.add(ReviewDeadline::new(1, BASE_TS + 300)); // approaching (< 1h)
        col.add(ReviewDeadline::new(2, BASE_TS + 7200)); // not approaching (> 1h)
        let approaching = col.approaching_at(BASE_TS, 3600);
        assert_eq!(approaching.len(), 1);
        assert_eq!(approaching[0].review_id, 1);
    }
}
