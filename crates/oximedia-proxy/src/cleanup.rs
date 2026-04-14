//! Proxy cleanup scheduling and expiration management.
//!
//! [`ProxyCleanupScheduler`] determines whether a proxy has exceeded its
//! configured maximum age and should be deleted or regenerated.

/// Scheduler for expiring and cleaning up stale proxy files.
#[derive(Debug, Clone)]
pub struct ProxyCleanupScheduler {
    /// Maximum age of a proxy in days before it is considered expired.
    max_age_days: u32,
}

impl ProxyCleanupScheduler {
    /// Create a new scheduler with the given maximum age in days.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_proxy::cleanup::ProxyCleanupScheduler;
    ///
    /// let sched = ProxyCleanupScheduler::new(30);
    /// assert!(!sched.is_expired(0, 86_400 * 29)); // 29 days — still valid
    /// assert!(sched.is_expired(0, 86_400 * 31));  // 31 days — expired
    /// ```
    #[must_use]
    pub fn new(max_age_days: u32) -> Self {
        Self { max_age_days }
    }

    /// Return whether a proxy created at `created_ts` (Unix timestamp, seconds)
    /// has expired relative to the current time `now_ts`.
    ///
    /// A proxy is expired when `(now_ts - created_ts) > max_age_days * 86400`.
    #[must_use]
    pub fn is_expired(&self, created_ts: u64, now_ts: u64) -> bool {
        let max_age_secs = self.max_age_days as u64 * 86_400;
        now_ts.saturating_sub(created_ts) > max_age_secs
    }

    /// Return the expiry timestamp for a proxy created at `created_ts`.
    ///
    /// The expiry timestamp is `created_ts + max_age_days * 86_400`.
    #[must_use]
    pub fn expiry_ts(&self, created_ts: u64) -> u64 {
        created_ts.saturating_add(self.max_age_days as u64 * 86_400)
    }

    /// Return the remaining lifetime in seconds for a proxy, or `None` if it
    /// has already expired.
    #[must_use]
    pub fn remaining_secs(&self, created_ts: u64, now_ts: u64) -> Option<u64> {
        let expiry = self.expiry_ts(created_ts);
        if now_ts >= expiry {
            None
        } else {
            Some(expiry - now_ts)
        }
    }

    /// Filter a list of `(id, created_ts)` pairs, returning only those that
    /// have expired given `now_ts`.
    #[must_use]
    pub fn filter_expired<'a, T: Copy>(
        &self,
        entries: &'a [(T, u64)],
        now_ts: u64,
    ) -> Vec<T> {
        entries
            .iter()
            .filter_map(|(id, ts)| {
                if self.is_expired(*ts, now_ts) {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Return the configured `max_age_days`.
    #[must_use]
    pub fn max_age_days(&self) -> u32 {
        self.max_age_days
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY: u64 = 86_400;

    #[test]
    fn not_expired_before_threshold() {
        let sched = ProxyCleanupScheduler::new(7);
        // Created at t=0, now=6 days → within threshold
        assert!(!sched.is_expired(0, DAY * 6));
    }

    #[test]
    fn not_expired_exactly_at_threshold() {
        let sched = ProxyCleanupScheduler::new(7);
        // Exactly 7 days → not expired (> not >=)
        assert!(!sched.is_expired(0, DAY * 7));
    }

    #[test]
    fn expired_after_threshold() {
        let sched = ProxyCleanupScheduler::new(7);
        assert!(sched.is_expired(0, DAY * 8));
    }

    #[test]
    fn expiry_ts_is_correct() {
        let sched = ProxyCleanupScheduler::new(30);
        assert_eq!(sched.expiry_ts(1_000_000), 1_000_000 + 30 * DAY);
    }

    #[test]
    fn remaining_secs_some_when_valid() {
        let sched = ProxyCleanupScheduler::new(10);
        let remaining = sched.remaining_secs(0, DAY * 5);
        assert_eq!(remaining, Some(DAY * 5));
    }

    #[test]
    fn remaining_secs_none_when_expired() {
        let sched = ProxyCleanupScheduler::new(10);
        assert_eq!(sched.remaining_secs(0, DAY * 11), None);
    }

    #[test]
    fn remaining_secs_none_exactly_at_expiry() {
        let sched = ProxyCleanupScheduler::new(5);
        assert_eq!(sched.remaining_secs(0, DAY * 5), None);
    }

    #[test]
    fn filter_expired_returns_expired_ids() {
        let sched = ProxyCleanupScheduler::new(7);
        let now = DAY * 10;
        let entries: Vec<(u32, u64)> = vec![
            (1, 0),          // created at day 0 — expired (10 days ago)
            (2, DAY * 8),    // created 2 days ago — not expired
            (3, DAY * 1),    // created 9 days ago — expired
        ];
        let expired = sched.filter_expired(&entries, now);
        assert!(expired.contains(&1));
        assert!(expired.contains(&3));
        assert!(!expired.contains(&2));
    }

    #[test]
    fn filter_expired_empty_input_returns_empty() {
        let sched = ProxyCleanupScheduler::new(14);
        let empty: Vec<(u32, u64)> = Vec::new();
        assert!(sched.filter_expired(&empty, DAY * 100).is_empty());
    }

    #[test]
    fn zero_max_age_expires_immediately() {
        let sched = ProxyCleanupScheduler::new(0);
        // now > created → expired
        assert!(sched.is_expired(0, 1));
    }

    #[test]
    fn max_age_days_accessor() {
        let sched = ProxyCleanupScheduler::new(42);
        assert_eq!(sched.max_age_days(), 42);
    }
}
