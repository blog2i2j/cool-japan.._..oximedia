//! High-level review comparison utilities.
//!
//! This module provides [`ReviewComparator`] which computes a structured diff
//! between two [`Review`] records — detecting changes in metadata (title, status,
//! assignee, etc.) and producing a `Vec<ReviewDiff>` describing each difference.
//!
//! ## Types
//!
//! * [`Review`]          — a lightweight snapshot of a review's observable state.
//! * [`ReviewDiff`]      — a single field-level change between two review snapshots.
//! * [`ReviewComparator`] — entry point; call [`ReviewComparator::diff`] to compare.
//!
//! ## Example
//!
//! ```rust
//! use oximedia_review::review_comparator::{Review, ReviewComparator, ReviewStatus};
//!
//! let v1 = Review {
//!     id: 1,
//!     title: "Scene 01 Grade".to_string(),
//!     status: ReviewStatus::InProgress,
//!     assignee: Some("alice@example.com".to_string()),
//!     priority: 3,
//!     comment_count: 5,
//!     version: 1,
//! };
//! let v2 = Review {
//!     id: 1,
//!     title: "Scene 01 Grade".to_string(),
//!     status: ReviewStatus::Approved,
//!     assignee: Some("alice@example.com".to_string()),
//!     priority: 3,
//!     comment_count: 5,
//!     version: 2,
//! };
//! let diffs = ReviewComparator::diff(&v1, &v2);
//! assert_eq!(diffs.len(), 2); // status + version changed
//! ```

// ─── Review ───────────────────────────────────────────────────────────────────

/// Current lifecycle state of a review.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReviewStatus {
    /// Review is in draft / not yet started.
    Draft,
    /// Active review is in progress.
    InProgress,
    /// Review has been approved.
    Approved,
    /// Review was rejected; author must revise.
    Rejected,
    /// Additional changes have been requested.
    ChangesRequested,
    /// Review is on hold.
    OnHold,
    /// Review was closed/cancelled.
    Closed,
}

impl std::fmt::Display for ReviewStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Draft => "Draft",
            Self::InProgress => "InProgress",
            Self::Approved => "Approved",
            Self::Rejected => "Rejected",
            Self::ChangesRequested => "ChangesRequested",
            Self::OnHold => "OnHold",
            Self::Closed => "Closed",
        };
        write!(f, "{s}")
    }
}

/// A lightweight snapshot of a review's observable state.
///
/// This is intentionally a plain data struct — not the full session object —
/// so that comparisons can be made between any two time-slices of a review
/// without requiring access to the live persistence layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Review {
    /// Unique review identifier.
    pub id: u64,
    /// Human-readable title.
    pub title: String,
    /// Current lifecycle status.
    pub status: ReviewStatus,
    /// Assigned reviewer e-mail, if any.
    pub assignee: Option<String>,
    /// Priority level (higher = more urgent).
    pub priority: u8,
    /// Number of comments attached to this review.
    pub comment_count: u32,
    /// Monotonically increasing version counter (incremented on every change).
    pub version: u64,
}

// ─── ReviewDiff ───────────────────────────────────────────────────────────────

/// A single field-level change between two review snapshots.
///
/// Each variant corresponds to one observable field on [`Review`].  When the
/// field value is the same in both snapshots, no `ReviewDiff` is emitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReviewDiff {
    /// The review title changed.
    TitleChanged {
        /// Value in the first snapshot.
        from: String,
        /// Value in the second snapshot.
        to: String,
    },
    /// The status changed.
    StatusChanged {
        /// Status in the first snapshot.
        from: ReviewStatus,
        /// Status in the second snapshot.
        to: ReviewStatus,
    },
    /// The assigned reviewer changed.
    AssigneeChanged {
        /// Assignee in the first snapshot (if any).
        from: Option<String>,
        /// Assignee in the second snapshot (if any).
        to: Option<String>,
    },
    /// The priority changed.
    PriorityChanged {
        /// Priority in the first snapshot.
        from: u8,
        /// Priority in the second snapshot.
        to: u8,
    },
    /// The number of comments changed.
    CommentCountChanged {
        /// Count in the first snapshot.
        from: u32,
        /// Count in the second snapshot.
        to: u32,
    },
    /// The internal version counter changed (always present when any other
    /// field changed, since version is bumped on every mutation).
    VersionChanged {
        /// Version in the first snapshot.
        from: u64,
        /// Version in the second snapshot.
        to: u64,
    },
}

impl std::fmt::Display for ReviewDiff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TitleChanged { from, to } => write!(f, "title: {from:?} → {to:?}"),
            Self::StatusChanged { from, to } => write!(f, "status: {from} → {to}"),
            Self::AssigneeChanged { from, to } => {
                let from_s = from.as_deref().unwrap_or("(none)");
                let to_s = to.as_deref().unwrap_or("(none)");
                write!(f, "assignee: {from_s} → {to_s}")
            }
            Self::PriorityChanged { from, to } => write!(f, "priority: {from} → {to}"),
            Self::CommentCountChanged { from, to } => write!(f, "comment_count: {from} → {to}"),
            Self::VersionChanged { from, to } => write!(f, "version: {from} → {to}"),
        }
    }
}

// ─── ReviewComparator ─────────────────────────────────────────────────────────

/// Compute field-level diffs between review snapshots.
pub struct ReviewComparator;

impl ReviewComparator {
    /// Produce a list of field-level differences between two review snapshots.
    ///
    /// The diff is computed over all observable fields on [`Review`].  If the
    /// two snapshots are identical (same ID, same field values) an empty vector
    /// is returned.
    ///
    /// Note: the `id` field is **not** diffed — both snapshots are assumed to
    /// represent the same logical review at different points in time.
    ///
    /// # Parameters
    ///
    /// * `v1` — first (earlier) snapshot
    /// * `v2` — second (later) snapshot
    ///
    /// # Returns
    ///
    /// A `Vec<ReviewDiff>` in field declaration order: title, status,
    /// assignee, priority, comment_count, version.
    #[must_use]
    pub fn diff(v1: &Review, v2: &Review) -> Vec<ReviewDiff> {
        let mut diffs = Vec::new();

        if v1.title != v2.title {
            diffs.push(ReviewDiff::TitleChanged {
                from: v1.title.clone(),
                to: v2.title.clone(),
            });
        }

        if v1.status != v2.status {
            diffs.push(ReviewDiff::StatusChanged {
                from: v1.status,
                to: v2.status,
            });
        }

        if v1.assignee != v2.assignee {
            diffs.push(ReviewDiff::AssigneeChanged {
                from: v1.assignee.clone(),
                to: v2.assignee.clone(),
            });
        }

        if v1.priority != v2.priority {
            diffs.push(ReviewDiff::PriorityChanged {
                from: v1.priority,
                to: v2.priority,
            });
        }

        if v1.comment_count != v2.comment_count {
            diffs.push(ReviewDiff::CommentCountChanged {
                from: v1.comment_count,
                to: v2.comment_count,
            });
        }

        if v1.version != v2.version {
            diffs.push(ReviewDiff::VersionChanged {
                from: v1.version,
                to: v2.version,
            });
        }

        diffs
    }

    /// Returns `true` if the two snapshots are field-for-field identical.
    #[must_use]
    pub fn is_equal(v1: &Review, v2: &Review) -> bool {
        Self::diff(v1, v2).is_empty()
    }

    /// Compute a summary string listing all changes in human-readable form.
    #[must_use]
    pub fn summary(v1: &Review, v2: &Review) -> String {
        let diffs = Self::diff(v1, v2);
        if diffs.is_empty() {
            return "no changes".to_string();
        }
        diffs
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("; ")
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn base_review() -> Review {
        Review {
            id: 1,
            title: "Cut Review".to_string(),
            status: ReviewStatus::InProgress,
            assignee: Some("alice@example.com".to_string()),
            priority: 5,
            comment_count: 10,
            version: 3,
        }
    }

    // ── identical reviews ─────────────────────────────────────────────────────

    #[test]
    fn diff_identical_returns_empty() {
        let r = base_review();
        let diffs = ReviewComparator::diff(&r, &r);
        assert!(diffs.is_empty());
    }

    #[test]
    fn is_equal_on_identical() {
        let r = base_review();
        assert!(ReviewComparator::is_equal(&r, &r));
    }

    // ── single-field changes ──────────────────────────────────────────────────

    #[test]
    fn diff_title_changed() {
        let v1 = base_review();
        let mut v2 = v1.clone();
        v2.title = "Final Cut".to_string();
        let diffs = ReviewComparator::diff(&v1, &v2);
        assert!(diffs
            .iter()
            .any(|d| matches!(d, ReviewDiff::TitleChanged { .. })));
    }

    #[test]
    fn diff_status_changed() {
        let v1 = base_review();
        let mut v2 = v1.clone();
        v2.status = ReviewStatus::Approved;
        let diffs = ReviewComparator::diff(&v1, &v2);
        assert!(diffs
            .iter()
            .any(|d| matches!(d, ReviewDiff::StatusChanged { .. })));
    }

    #[test]
    fn diff_assignee_changed_to_none() {
        let v1 = base_review();
        let mut v2 = v1.clone();
        v2.assignee = None;
        let diffs = ReviewComparator::diff(&v1, &v2);
        assert!(diffs
            .iter()
            .any(|d| matches!(d, ReviewDiff::AssigneeChanged { .. })));
    }

    #[test]
    fn diff_priority_changed() {
        let v1 = base_review();
        let mut v2 = v1.clone();
        v2.priority = 1;
        let diffs = ReviewComparator::diff(&v1, &v2);
        assert!(diffs
            .iter()
            .any(|d| matches!(d, ReviewDiff::PriorityChanged { from: 5, to: 1 })));
    }

    #[test]
    fn diff_comment_count_changed() {
        let v1 = base_review();
        let mut v2 = v1.clone();
        v2.comment_count = 20;
        let diffs = ReviewComparator::diff(&v1, &v2);
        assert!(diffs
            .iter()
            .any(|d| matches!(d, ReviewDiff::CommentCountChanged { from: 10, to: 20 })));
    }

    #[test]
    fn diff_version_changed() {
        let v1 = base_review();
        let mut v2 = v1.clone();
        v2.version = 4;
        let diffs = ReviewComparator::diff(&v1, &v2);
        assert!(diffs
            .iter()
            .any(|d| matches!(d, ReviewDiff::VersionChanged { from: 3, to: 4 })));
    }

    // ── multi-field changes ───────────────────────────────────────────────────

    #[test]
    fn diff_multiple_fields_changed() {
        let v1 = base_review();
        let mut v2 = v1.clone();
        v2.status = ReviewStatus::Approved;
        v2.version = 4;
        let diffs = ReviewComparator::diff(&v1, &v2);
        assert_eq!(diffs.len(), 2);
    }

    #[test]
    fn diff_all_fields_changed() {
        let v1 = base_review();
        let v2 = Review {
            id: 1, // same id
            title: "New Title".to_string(),
            status: ReviewStatus::Closed,
            assignee: None,
            priority: 1,
            comment_count: 0,
            version: 99,
        };
        let diffs = ReviewComparator::diff(&v1, &v2);
        assert_eq!(diffs.len(), 6);
    }

    // ── summary ───────────────────────────────────────────────────────────────

    #[test]
    fn summary_no_changes() {
        let r = base_review();
        assert_eq!(ReviewComparator::summary(&r, &r), "no changes");
    }

    #[test]
    fn summary_with_changes_contains_field_name() {
        let v1 = base_review();
        let mut v2 = v1.clone();
        v2.status = ReviewStatus::Approved;
        let s = ReviewComparator::summary(&v1, &v2);
        assert!(s.contains("status"), "summary='{s}'");
    }

    // ── ReviewDiff::Display ───────────────────────────────────────────────────

    #[test]
    fn diff_display_status() {
        let d = ReviewDiff::StatusChanged {
            from: ReviewStatus::Draft,
            to: ReviewStatus::Approved,
        };
        let s = d.to_string();
        assert!(s.contains("Draft"), "s='{s}'");
        assert!(s.contains("Approved"), "s='{s}'");
    }
}
