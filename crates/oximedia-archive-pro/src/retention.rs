//! Archive retention management module.
//!
//! Provides policies for how long archived assets must be kept, tools for
//! tracking individual asset retention records, and a scheduler that surfaces
//! assets due for deletion or annual review.

use serde::{Deserialize, Serialize};

/// Seconds per year (365 × 24 × 3600).
const SECS_PER_YEAR: u64 = 365 * 24 * 3600;

/// Seconds per day.
const SECS_PER_DAY: u64 = 24 * 3600;

/// A retention policy that describes how long an asset must be kept.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetentionPolicy {
    /// Asset must be kept forever.
    Permanent,
    /// Asset expires N years after its creation date.
    YearsFromCreation(u32),
    /// Asset expires N years after the last access date.
    YearsFromLastAccess(u32),
    /// Asset expires at a specific Unix timestamp (seconds).
    UntilDate(u64),
    /// Asset has no fixed expiry but must be reviewed once per year.
    ReviewAnnually,
}

impl RetentionPolicy {
    /// Compute the Unix timestamp at which the asset expires, if applicable.
    ///
    /// Returns `None` for `Permanent` and `ReviewAnnually` policies because they
    /// have no hard expiry date.
    #[must_use]
    pub fn expires_at(&self, created_at: u64, last_accessed: u64) -> Option<u64> {
        match self {
            Self::Permanent | Self::ReviewAnnually => None,
            Self::YearsFromCreation(years) => {
                Some(created_at.saturating_add(*years as u64 * SECS_PER_YEAR))
            }
            Self::YearsFromLastAccess(years) => {
                Some(last_accessed.saturating_add(*years as u64 * SECS_PER_YEAR))
            }
            Self::UntilDate(ts) => Some(*ts),
        }
    }
}

/// A retention record tracking a single archived asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionRecord {
    /// Unique identifier of the archived asset
    pub asset_id: String,
    /// Applicable retention policy
    pub policy: RetentionPolicy,
    /// Unix timestamp when the asset was created/ingested
    pub created_at: u64,
    /// Unix timestamp of the most recent access
    pub last_accessed: u64,
    /// Unix timestamp when the next annual review is due (if applicable)
    pub review_due: Option<u64>,
}

impl RetentionRecord {
    /// Create a new `RetentionRecord` with `last_accessed` equal to `created_at`.
    #[must_use]
    pub fn new(id: &str, policy: RetentionPolicy, created_at: u64) -> Self {
        let review_due = match &policy {
            RetentionPolicy::ReviewAnnually => Some(created_at + SECS_PER_YEAR),
            _ => None,
        };
        Self {
            asset_id: id.to_owned(),
            policy,
            created_at,
            last_accessed: created_at,
            review_due,
        }
    }

    /// Returns `true` if the asset's retention period has expired as of `now`.
    #[must_use]
    pub fn is_expired(&self, now: u64) -> bool {
        match self.policy.expires_at(self.created_at, self.last_accessed) {
            Some(expiry) => now >= expiry,
            None => false,
        }
    }

    /// Number of days until expiry, or `None` for non-expiring policies.
    ///
    /// Returns a negative value if already expired.
    #[allow(clippy::cast_possible_wrap)]
    #[must_use]
    pub fn days_until_expiry(&self, now: u64) -> Option<i64> {
        self.policy
            .expires_at(self.created_at, self.last_accessed)
            .map(|expiry| {
                let diff_secs = expiry as i64 - now as i64;
                diff_secs / SECS_PER_DAY as i64
            })
    }

    /// Update the last-accessed timestamp and refresh `YearsFromLastAccess` expiry.
    pub fn mark_accessed(&mut self, now: u64) {
        self.last_accessed = now;
        // Advance the review due date for ReviewAnnually policies
        if let RetentionPolicy::ReviewAnnually = self.policy {
            self.review_due = Some(now + SECS_PER_YEAR);
        }
    }
}

/// A scheduler that manages a collection of retention records.
#[derive(Debug, Default)]
pub struct RetentionScheduler {
    /// All tracked retention records
    pub records: Vec<RetentionRecord>,
}

impl RetentionScheduler {
    /// Create a new empty `RetentionScheduler`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a retention record to the scheduler.
    pub fn add(&mut self, r: RetentionRecord) {
        self.records.push(r);
    }

    /// Return references to all records whose retention period has expired at `now`.
    #[must_use]
    pub fn due_for_deletion(&self, now: u64) -> Vec<&RetentionRecord> {
        self.records.iter().filter(|r| r.is_expired(now)).collect()
    }

    /// Return references to all records whose annual review is due at or before `now`.
    #[must_use]
    pub fn due_for_review(&self, now: u64) -> Vec<&RetentionRecord> {
        self.records
            .iter()
            .filter(|r| r.review_due.is_some_and(|due| now >= due))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const T0: u64 = 1_700_000_000; // arbitrary base timestamp

    fn year_secs(years: u64) -> u64 {
        years * SECS_PER_YEAR
    }

    #[test]
    fn test_permanent_never_expires() {
        let r = RetentionRecord::new("asset-1", RetentionPolicy::Permanent, T0);
        assert!(!r.is_expired(T0 + year_secs(100)));
    }

    #[test]
    fn test_years_from_creation_expires() {
        let r = RetentionRecord::new("asset-2", RetentionPolicy::YearsFromCreation(5), T0);
        let five_years_later = T0 + year_secs(5);
        assert!(r.is_expired(five_years_later));
    }

    #[test]
    fn test_years_from_creation_not_expired() {
        let r = RetentionRecord::new("asset-3", RetentionPolicy::YearsFromCreation(5), T0);
        assert!(!r.is_expired(T0 + year_secs(4)));
    }

    #[test]
    fn test_years_from_last_access_uses_last_accessed() {
        let mut r = RetentionRecord::new("asset-4", RetentionPolicy::YearsFromLastAccess(2), T0);
        let later = T0 + year_secs(1);
        r.mark_accessed(later);
        // Should not expire 1 year after access (only 2 years policy)
        assert!(!r.is_expired(later + year_secs(1)));
        // Should expire 2 years after last access
        assert!(r.is_expired(later + year_secs(2)));
    }

    #[test]
    fn test_until_date_expires() {
        let expiry = T0 + 90 * SECS_PER_DAY;
        let r = RetentionRecord::new("asset-5", RetentionPolicy::UntilDate(expiry), T0);
        assert!(!r.is_expired(expiry - 1));
        assert!(r.is_expired(expiry));
    }

    #[test]
    fn test_review_annually_no_expiry() {
        let r = RetentionRecord::new("asset-6", RetentionPolicy::ReviewAnnually, T0);
        assert!(!r.is_expired(T0 + year_secs(50)));
    }

    #[test]
    fn test_review_annually_has_review_due() {
        let r = RetentionRecord::new("asset-7", RetentionPolicy::ReviewAnnually, T0);
        assert!(r.review_due.is_some());
        assert_eq!(
            r.review_due.expect("operation should succeed"),
            T0 + SECS_PER_YEAR
        );
    }

    #[test]
    fn test_mark_accessed_updates_timestamp() {
        let mut r = RetentionRecord::new("asset-8", RetentionPolicy::Permanent, T0);
        r.mark_accessed(T0 + 1000);
        assert_eq!(r.last_accessed, T0 + 1000);
    }

    #[test]
    fn test_mark_accessed_advances_review_due() {
        let mut r = RetentionRecord::new("asset-9", RetentionPolicy::ReviewAnnually, T0);
        let new_access = T0 + 500;
        r.mark_accessed(new_access);
        assert_eq!(
            r.review_due.expect("operation should succeed"),
            new_access + SECS_PER_YEAR
        );
    }

    #[test]
    fn test_days_until_expiry_positive() {
        let r = RetentionRecord::new("asset-10", RetentionPolicy::YearsFromCreation(1), T0);
        let days = r.days_until_expiry(T0).expect("operation should succeed");
        assert_eq!(days, 365);
    }

    #[test]
    fn test_days_until_expiry_negative_when_expired() {
        let r = RetentionRecord::new("asset-11", RetentionPolicy::YearsFromCreation(1), T0);
        let days = r
            .days_until_expiry(T0 + year_secs(2))
            .expect("operation should succeed");
        assert!(days < 0);
    }

    #[test]
    fn test_days_until_expiry_none_for_permanent() {
        let r = RetentionRecord::new("asset-12", RetentionPolicy::Permanent, T0);
        assert!(r.days_until_expiry(T0).is_none());
    }

    #[test]
    fn test_scheduler_due_for_deletion() {
        let mut sched = RetentionScheduler::new();
        sched.add(RetentionRecord::new(
            "a",
            RetentionPolicy::YearsFromCreation(1),
            T0,
        ));
        sched.add(RetentionRecord::new(
            "b",
            RetentionPolicy::YearsFromCreation(10),
            T0,
        ));
        let now = T0 + year_secs(2);
        let expired = sched.due_for_deletion(now);
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].asset_id, "a");
    }

    #[test]
    fn test_scheduler_due_for_review() {
        let mut sched = RetentionScheduler::new();
        sched.add(RetentionRecord::new(
            "x",
            RetentionPolicy::ReviewAnnually,
            T0,
        ));
        sched.add(RetentionRecord::new("y", RetentionPolicy::Permanent, T0));
        let now = T0 + year_secs(1) + 1;
        let reviews = sched.due_for_review(now);
        assert_eq!(reviews.len(), 1);
        assert_eq!(reviews[0].asset_id, "x");
    }

    #[test]
    fn test_scheduler_empty() {
        let sched = RetentionScheduler::new();
        assert!(sched.due_for_deletion(T0).is_empty());
        assert!(sched.due_for_review(T0).is_empty());
    }
}
