//! Rights expiry alerting.
//!
//! [`RightsExpiryAlerter`] scans a slice of [`Right`] records and returns
//! alerts for rights that are about to expire (within `warn_days` days from
//! `now_ts`).

/// Minimal rights record for expiry checking.
#[derive(Debug, Clone)]
pub struct Right {
    /// Unique identifier for this right.
    pub id: u64,
    /// Human-readable name or description.
    pub name: String,
    /// Unix timestamp (seconds) when the right expires.
    pub expires_ts: u64,
}

impl Right {
    /// Create a new right record.
    pub fn new(id: u64, name: impl Into<String>, expires_ts: u64) -> Self {
        Self {
            id,
            name: name.into(),
            expires_ts,
        }
    }
}

/// An alert for a right that is about to expire.
#[derive(Debug, Clone)]
pub struct ExpiryAlert {
    /// ID of the right that is expiring.
    pub rights_id: u64,
    /// Name of the right.
    pub name: String,
    /// Unix timestamp when the right expires.
    pub expires_ts: u64,
    /// Seconds remaining until expiry (0 if already expired).
    pub seconds_remaining: u64,
    /// Whether the right has already expired at `now_ts`.
    pub already_expired: bool,
}

/// Checks a set of rights for upcoming or past expiry.
pub struct RightsExpiryAlerter;

impl RightsExpiryAlerter {
    /// Check `rights` for expiry alerts.
    ///
    /// An alert is generated for every right whose expiry timestamp falls
    /// within `now_ts + warn_days * 86_400` seconds (or that has already
    /// expired).
    ///
    /// # Parameters
    /// - `rights` – slice of [`Right`] records to inspect.
    /// - `now_ts` – current Unix timestamp in seconds.
    /// - `warn_days` – number of days before expiry to start warning.
    pub fn check(rights: &[Right], now_ts: u64, warn_days: u64) -> Vec<ExpiryAlert> {
        let warn_window_secs = warn_days.saturating_mul(86_400);
        let cutoff = now_ts.saturating_add(warn_window_secs);

        rights
            .iter()
            .filter_map(|r| {
                if r.expires_ts <= cutoff {
                    let already_expired = r.expires_ts <= now_ts;
                    let seconds_remaining = if already_expired {
                        0
                    } else {
                        r.expires_ts - now_ts
                    };
                    Some(ExpiryAlert {
                        rights_id: r.id,
                        name: r.name.clone(),
                        expires_ts: r.expires_ts,
                        seconds_remaining,
                        already_expired,
                    })
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_700_000_000; // arbitrary fixed timestamp

    #[test]
    fn test_no_alerts_far_future() {
        let rights = vec![Right::new(1, "License A", NOW + 90 * 86_400)];
        let alerts = RightsExpiryAlerter::check(&rights, NOW, 30);
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_alert_within_warn_days() {
        let rights = vec![Right::new(1, "License B", NOW + 10 * 86_400)];
        let alerts = RightsExpiryAlerter::check(&rights, NOW, 30);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].rights_id, 1);
        assert!(!alerts[0].already_expired);
        assert_eq!(alerts[0].seconds_remaining, 10 * 86_400);
    }

    #[test]
    fn test_alert_already_expired() {
        let rights = vec![Right::new(2, "Old License", NOW - 3_600)];
        let alerts = RightsExpiryAlerter::check(&rights, NOW, 7);
        assert_eq!(alerts.len(), 1);
        assert!(alerts[0].already_expired);
        assert_eq!(alerts[0].seconds_remaining, 0);
    }

    #[test]
    fn test_mixed_rights() {
        let rights = vec![
            Right::new(1, "Far future", NOW + 365 * 86_400),
            Right::new(2, "Soon", NOW + 5 * 86_400),
            Right::new(3, "Expired", NOW - 1),
        ];
        let alerts = RightsExpiryAlerter::check(&rights, NOW, 14);
        assert_eq!(alerts.len(), 2);
        let ids: Vec<u64> = alerts.iter().map(|a| a.rights_id).collect();
        assert!(ids.contains(&2));
        assert!(ids.contains(&3));
    }

    #[test]
    fn test_zero_warn_days_only_expired() {
        let rights = vec![
            Right::new(1, "Expired", NOW - 1),
            Right::new(2, "Active", NOW + 1),
        ];
        let alerts = RightsExpiryAlerter::check(&rights, NOW, 0);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].rights_id, 1);
    }

    #[test]
    fn test_alert_at_exact_expiry_boundary() {
        // expires_ts == now_ts → already expired
        let rights = vec![Right::new(5, "At boundary", NOW)];
        let alerts = RightsExpiryAlerter::check(&rights, NOW, 0);
        assert_eq!(alerts.len(), 1);
        assert!(alerts[0].already_expired);
        assert_eq!(alerts[0].seconds_remaining, 0);
    }

    #[test]
    fn test_seconds_remaining_correct() {
        let rights = vec![Right::new(6, "Soon", NOW + 3_600)];
        let alerts = RightsExpiryAlerter::check(&rights, NOW, 1);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].seconds_remaining, 3_600);
        assert!(!alerts[0].already_expired);
    }

    #[test]
    fn test_empty_rights_slice() {
        let alerts = RightsExpiryAlerter::check(&[], NOW, 30);
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_right_name_preserved_in_alert() {
        let rights = vec![Right::new(7, "My Important License", NOW + 100)];
        let alerts = RightsExpiryAlerter::check(&rights, NOW, 1);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].name, "My Important License");
    }

    #[test]
    fn test_alert_expires_ts_matches_right() {
        let expire_at = NOW + 5_000;
        let rights = vec![Right::new(8, "Check TS", expire_at)];
        let alerts = RightsExpiryAlerter::check(&rights, NOW, 1);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].expires_ts, expire_at);
    }
}
