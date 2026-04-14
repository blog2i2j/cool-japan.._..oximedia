//! Raft leader lease implementation.
//!
//! A leader lease prevents followers from becoming a leader while the current
//! leader's lease is valid.  This avoids unnecessary elections when the leader
//! is still reachable.

/// A time-bounded lease held by the Raft leader.
///
/// The lease is defined by:
/// - A `duration_ms` that specifies how long a single grant lasts.
/// - A `granted_at` timestamp (milliseconds since an arbitrary epoch, e.g.
///   Unix epoch) recording when the lease was last renewed.
///
/// The lease is considered valid as long as `ts < granted_at + duration_ms`.
#[derive(Debug, Clone)]
pub struct LeaderLease {
    /// Lease duration in milliseconds.
    pub duration_ms: u64,
    /// Timestamp (ms) when the lease was last granted / renewed.
    /// `None` means the lease has never been granted.
    pub granted_at: Option<u64>,
}

impl LeaderLease {
    /// Create a new `LeaderLease` with the given duration.
    ///
    /// The lease starts in the un-granted state and must be renewed before it
    /// can be considered valid.
    #[must_use]
    pub fn new(duration_ms: u64) -> Self {
        Self {
            duration_ms,
            granted_at: None,
        }
    }

    /// Renew (or initially grant) the lease at timestamp `ts` (milliseconds).
    ///
    /// After calling this, [`Self::is_valid`] will return `true` for any `query_ts`
    /// in `[ts, ts + duration_ms)`.
    pub fn renew(&mut self, ts: u64) {
        self.granted_at = Some(ts);
    }

    /// Returns `true` if the lease is valid at query timestamp `ts` (ms).
    ///
    /// A lease is valid when it has been granted and `ts < granted_at + duration_ms`.
    #[must_use]
    pub fn is_valid(&self, ts: u64) -> bool {
        match self.granted_at {
            None => false,
            Some(granted) => {
                // Use saturating add to avoid u64 overflow
                ts < granted.saturating_add(self.duration_ms)
            }
        }
    }

    /// Remaining validity time in milliseconds, or `0` if expired / not granted.
    #[must_use]
    pub fn remaining_ms(&self, ts: u64) -> u64 {
        match self.granted_at {
            None => 0,
            Some(granted) => {
                let expiry = granted.saturating_add(self.duration_ms);
                expiry.saturating_sub(ts)
            }
        }
    }

    /// Invalidate the lease immediately (e.g., on leader step-down).
    pub fn invalidate(&mut self) {
        self.granted_at = None;
    }

    /// Return the expiry timestamp (ms), or `None` if not yet granted.
    #[must_use]
    pub fn expiry(&self) -> Option<u64> {
        self.granted_at.map(|g| g.saturating_add(self.duration_ms))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_lease_is_not_valid() {
        let lease = LeaderLease::new(5000);
        assert!(!lease.is_valid(0));
        assert!(!lease.is_valid(1000));
    }

    #[test]
    fn test_renew_makes_lease_valid() {
        let mut lease = LeaderLease::new(5000);
        lease.renew(10_000);
        assert!(lease.is_valid(10_000));
        assert!(lease.is_valid(14_999));
    }

    #[test]
    fn test_lease_expires() {
        let mut lease = LeaderLease::new(5000);
        lease.renew(10_000);
        // Exactly at expiry boundary (10_000 + 5_000 = 15_000) → expired
        assert!(!lease.is_valid(15_000));
        assert!(!lease.is_valid(16_000));
    }

    #[test]
    fn test_renew_extends_lease() {
        let mut lease = LeaderLease::new(3000);
        lease.renew(0);
        assert!(lease.is_valid(2000));
        // Renew again at ts=2000
        lease.renew(2000);
        assert!(lease.is_valid(4000));
        assert!(!lease.is_valid(5001));
    }

    #[test]
    fn test_remaining_ms_before_expiry() {
        let mut lease = LeaderLease::new(10_000);
        lease.renew(0);
        assert_eq!(lease.remaining_ms(3_000), 7_000);
    }

    #[test]
    fn test_remaining_ms_after_expiry() {
        let mut lease = LeaderLease::new(5_000);
        lease.renew(0);
        assert_eq!(lease.remaining_ms(6_000), 0);
    }

    #[test]
    fn test_remaining_ms_not_granted() {
        let lease = LeaderLease::new(5_000);
        assert_eq!(lease.remaining_ms(0), 0);
    }

    #[test]
    fn test_invalidate_clears_lease() {
        let mut lease = LeaderLease::new(5_000);
        lease.renew(0);
        assert!(lease.is_valid(1_000));
        lease.invalidate();
        assert!(!lease.is_valid(1_000));
    }

    #[test]
    fn test_expiry_none_before_grant() {
        let lease = LeaderLease::new(5_000);
        assert!(lease.expiry().is_none());
    }

    #[test]
    fn test_expiry_some_after_grant() {
        let mut lease = LeaderLease::new(5_000);
        lease.renew(10_000);
        assert_eq!(lease.expiry(), Some(15_000));
    }

    #[test]
    fn test_overflow_safe() {
        let mut lease = LeaderLease::new(u64::MAX);
        lease.renew(1);
        // With duration = u64::MAX, expiry = 1 + u64::MAX → saturates at u64::MAX
        // is_valid(u64::MAX - 1) should be true (ts < u64::MAX)
        assert!(lease.is_valid(u64::MAX - 1));
    }

    #[test]
    fn test_zero_duration_expires_immediately() {
        let mut lease = LeaderLease::new(0);
        lease.renew(100);
        // ts < 100 + 0 = 100: only ts < 100 is valid
        assert!(lease.is_valid(99));
        assert!(!lease.is_valid(100));
    }
}
