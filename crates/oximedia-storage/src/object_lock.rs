//! Object lock (WORM storage) — compliance-mode and governance-mode locking.
//!
//! Object locking prevents objects from being deleted or overwritten for a
//! specified retention period.  Two modes are supported:
//!
//! - **Compliance mode**: once set, the lock *cannot* be shortened or removed
//!   by any user, including administrators.  This mirrors AWS S3 Object Lock
//!   compliance mode and SEC Rule 17a-4(f) requirements.
//!
//! - **Governance mode**: administrators with elevated permissions may remove
//!   or shorten the lock, but regular users cannot.  Suitable for internal
//!   data-management policies that may need adjustment.
//!
//! This module provides a pure-Rust, in-process lock registry.  Storage
//! backends that support native locking (e.g. AWS S3 Object Lock) should
//! delegate to their provider SDK; this registry acts as a local cache /
//! enforcement layer for provider-agnostic code paths.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use chrono::{DateTime, Duration, Utc};
use std::collections::HashMap;

// ─── LockMode ─────────────────────────────────────────────────────────────────

/// The locking mode applied to an object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LockMode {
    /// Compliance mode — lock cannot be removed or shortened by any user.
    Compliance,
    /// Governance mode — administrators can remove or shorten the lock.
    Governance,
}

impl LockMode {
    /// Returns a human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Compliance => "COMPLIANCE",
            Self::Governance => "GOVERNANCE",
        }
    }
}

// ─── LegalHold ────────────────────────────────────────────────────────────────

/// A legal hold placed on an object independent of its retention period.
///
/// While a legal hold is active the object cannot be deleted regardless of
/// any retention settings.  Legal holds must be explicitly removed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegalHold {
    /// The reason or case identifier for this legal hold.
    pub reason: String,
    /// When the hold was applied.
    pub placed_at: DateTime<Utc>,
}

impl LegalHold {
    /// Create a new legal hold.
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
            placed_at: Utc::now(),
        }
    }
}

// ─── ObjectLockRecord ─────────────────────────────────────────────────────────

/// A complete lock record for a single object key (+ optional version).
#[derive(Debug, Clone)]
pub struct ObjectLockRecord {
    /// Object key.
    pub key: String,
    /// Optional version ID.
    pub version_id: Option<String>,
    /// Locking mode.
    pub mode: LockMode,
    /// When the retention period expires (UTC).
    pub retain_until: DateTime<Utc>,
    /// Optional legal hold.
    pub legal_hold: Option<LegalHold>,
}

impl ObjectLockRecord {
    /// Returns `true` if the retention period is still active at `now`.
    pub fn is_retention_active(&self, now: DateTime<Utc>) -> bool {
        now < self.retain_until
    }

    /// Returns `true` if the object is protected from deletion (retention OR
    /// legal hold).
    pub fn is_protected(&self, now: DateTime<Utc>) -> bool {
        self.is_retention_active(now) || self.legal_hold.is_some()
    }
}

// ─── ObjectLockError ──────────────────────────────────────────────────────────

/// Errors from the object lock registry.
#[derive(Debug, thiserror::Error)]
pub enum ObjectLockError {
    /// The object is protected by a lock and cannot be deleted.
    #[error("object '{key}' is protected: {reason}")]
    Protected {
        /// Object key.
        key: String,
        /// Reason string (retention / legal hold).
        reason: String,
    },
    /// A compliance-mode lock cannot be shortened.
    #[error("compliance-mode lock on '{key}' cannot be modified")]
    ComplianceLock {
        /// Object key.
        key: String,
    },
    /// The requested retain-until date is in the past.
    #[error("retain_until must be in the future")]
    RetainUntilInPast,
    /// No lock record found for the key.
    #[error("no lock record for key '{0}'")]
    NotFound(String),
}

// ─── ObjectLockRegistry ───────────────────────────────────────────────────────

/// In-process registry of object lock records.
///
/// Thread-safety is the caller's responsibility; wrap in a `Mutex` or
/// `RwLock` when shared across threads.
#[derive(Debug, Default)]
pub struct ObjectLockRegistry {
    records: HashMap<String, ObjectLockRecord>,
}

impl ObjectLockRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    // ── Key helper ────────────────────────────────────────────────────────────

    fn record_key(key: &str, version_id: Option<&str>) -> String {
        match version_id {
            Some(v) => format!("{key}\x00{v}"),
            None => key.to_string(),
        }
    }

    // ── Locking ───────────────────────────────────────────────────────────────

    /// Apply a retention lock to an object.
    ///
    /// If a lock already exists in **governance** mode it can be overwritten.
    /// A **compliance**-mode lock can only have its `retain_until` extended,
    /// not shortened.
    ///
    /// # Errors
    ///
    /// - [`ObjectLockError::RetainUntilInPast`] if `retain_until` ≤ now.
    /// - [`ObjectLockError::ComplianceLock`] if attempting to shorten a
    ///   compliance-mode lock.
    pub fn lock(
        &mut self,
        key: &str,
        version_id: Option<&str>,
        mode: LockMode,
        retain_until: DateTime<Utc>,
    ) -> Result<(), ObjectLockError> {
        let now = Utc::now();
        if retain_until <= now {
            return Err(ObjectLockError::RetainUntilInPast);
        }
        let rk = Self::record_key(key, version_id);
        if let Some(existing) = self.records.get(&rk) {
            if existing.mode == LockMode::Compliance && retain_until < existing.retain_until {
                return Err(ObjectLockError::ComplianceLock {
                    key: key.to_string(),
                });
            }
        }
        self.records.insert(
            rk,
            ObjectLockRecord {
                key: key.to_string(),
                version_id: version_id.map(str::to_string),
                mode,
                retain_until,
                legal_hold: self
                    .records
                    .get(&Self::record_key(key, version_id))
                    .and_then(|r| r.legal_hold.clone()),
            },
        );
        Ok(())
    }

    /// Attempt to delete an object, enforcing lock constraints.
    ///
    /// # Errors
    ///
    /// Returns [`ObjectLockError::Protected`] if the object is locked or has an
    /// active legal hold.
    pub fn check_delete(
        &self,
        key: &str,
        version_id: Option<&str>,
        is_admin: bool,
    ) -> Result<(), ObjectLockError> {
        let rk = Self::record_key(key, version_id);
        let Some(record) = self.records.get(&rk) else {
            return Ok(()); // No lock — deletion is permitted.
        };
        let now = Utc::now();
        // Legal hold always blocks deletion regardless of mode or admin status.
        if record.legal_hold.is_some() {
            return Err(ObjectLockError::Protected {
                key: key.to_string(),
                reason: "legal hold is active".to_string(),
            });
        }
        if record.is_retention_active(now) {
            match record.mode {
                LockMode::Compliance => {
                    return Err(ObjectLockError::Protected {
                        key: key.to_string(),
                        reason: format!(
                            "compliance-mode lock expires {}",
                            record.retain_until.format("%Y-%m-%dT%H:%M:%SZ")
                        ),
                    });
                }
                LockMode::Governance => {
                    if !is_admin {
                        return Err(ObjectLockError::Protected {
                            key: key.to_string(),
                            reason: "governance-mode lock requires admin permission".to_string(),
                        });
                    }
                    // Admin bypass — allowed.
                }
            }
        }
        Ok(())
    }

    /// Place a legal hold on an object.
    ///
    /// Creates a lock record with a far-future retain_until if none exists.
    pub fn place_legal_hold(
        &mut self,
        key: &str,
        version_id: Option<&str>,
        reason: impl Into<String>,
    ) {
        let rk = Self::record_key(key, version_id);
        let hold = LegalHold::new(reason);
        if let Some(record) = self.records.get_mut(&rk) {
            record.legal_hold = Some(hold);
        } else {
            // Create a synthetic record so the hold can be tracked.
            self.records.insert(
                rk,
                ObjectLockRecord {
                    key: key.to_string(),
                    version_id: version_id.map(str::to_string),
                    mode: LockMode::Governance,
                    // Far-future retain-until so the record is kept.
                    retain_until: Utc::now() + Duration::days(36500),
                    legal_hold: Some(hold),
                },
            );
        }
    }

    /// Remove a legal hold from an object.
    pub fn release_legal_hold(&mut self, key: &str, version_id: Option<&str>) {
        let rk = Self::record_key(key, version_id);
        if let Some(record) = self.records.get_mut(&rk) {
            record.legal_hold = None;
        }
    }

    /// Retrieve the lock record for an object (if any).
    pub fn get_lock(&self, key: &str, version_id: Option<&str>) -> Option<&ObjectLockRecord> {
        let rk = Self::record_key(key, version_id);
        self.records.get(&rk)
    }

    /// Number of lock records currently held.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns `true` when no records are stored.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn future(secs: i64) -> DateTime<Utc> {
        Utc::now() + Duration::seconds(secs)
    }

    fn past(secs: i64) -> DateTime<Utc> {
        Utc::now() - Duration::seconds(secs)
    }

    #[test]
    fn test_lock_and_get() {
        let mut reg = ObjectLockRegistry::new();
        reg.lock("key/obj", None, LockMode::Compliance, future(3600))
            .expect("lock");
        let record = reg.get_lock("key/obj", None).expect("found");
        assert_eq!(record.mode, LockMode::Compliance);
    }

    #[test]
    fn test_lock_with_past_date_fails() {
        let mut reg = ObjectLockRegistry::new();
        let err = reg.lock("key", None, LockMode::Governance, past(10));
        assert!(matches!(err, Err(ObjectLockError::RetainUntilInPast)));
    }

    #[test]
    fn test_check_delete_no_lock_succeeds() {
        let reg = ObjectLockRegistry::new();
        assert!(reg.check_delete("no_lock", None, false).is_ok());
    }

    #[test]
    fn test_check_delete_compliance_lock_blocks_all() {
        let mut reg = ObjectLockRegistry::new();
        reg.lock("obj", None, LockMode::Compliance, future(3600))
            .expect("lock");
        assert!(reg.check_delete("obj", None, false).is_err());
        assert!(reg.check_delete("obj", None, true).is_err());
    }

    #[test]
    fn test_check_delete_governance_lock_blocks_non_admin() {
        let mut reg = ObjectLockRegistry::new();
        reg.lock("obj", None, LockMode::Governance, future(3600))
            .expect("lock");
        assert!(reg.check_delete("obj", None, false).is_err());
    }

    #[test]
    fn test_check_delete_governance_lock_allows_admin() {
        let mut reg = ObjectLockRegistry::new();
        reg.lock("obj", None, LockMode::Governance, future(3600))
            .expect("lock");
        assert!(reg.check_delete("obj", None, true).is_ok());
    }

    #[test]
    fn test_legal_hold_blocks_deletion_even_for_admin() {
        let mut reg = ObjectLockRegistry::new();
        reg.place_legal_hold("obj", None, "litigation");
        assert!(reg.check_delete("obj", None, true).is_err());
    }

    #[test]
    fn test_release_legal_hold_allows_deletion() {
        let mut reg = ObjectLockRegistry::new();
        reg.place_legal_hold("obj", None, "audit");
        reg.release_legal_hold("obj", None);
        // After releasing hold, check if deletion is permitted (expired retention)
        // The synthetic record has a far-future retain_until, so admin bypass needed.
        assert!(reg.check_delete("obj", None, true).is_ok());
    }

    #[test]
    fn test_compliance_lock_cannot_be_shortened() {
        let mut reg = ObjectLockRegistry::new();
        let t1 = future(7200); // 2 hours
        let t_shorter = future(3600); // 1 hour
        reg.lock("obj", None, LockMode::Compliance, t1)
            .expect("first lock");
        let err = reg.lock("obj", None, LockMode::Compliance, t_shorter);
        assert!(matches!(err, Err(ObjectLockError::ComplianceLock { .. })));
    }

    #[test]
    fn test_compliance_lock_can_be_extended() {
        let mut reg = ObjectLockRegistry::new();
        reg.lock("obj", None, LockMode::Compliance, future(3600))
            .expect("lock");
        // Extending is allowed
        reg.lock("obj", None, LockMode::Compliance, future(7200))
            .expect("extend");
        let record = reg.get_lock("obj", None).expect("found");
        assert!(record.retain_until > future(3601));
    }

    #[test]
    fn test_version_id_isolation() {
        let mut reg = ObjectLockRegistry::new();
        reg.lock("obj", Some("v1"), LockMode::Compliance, future(3600))
            .expect("lock v1");
        // v2 should not be locked
        assert!(reg.check_delete("obj", Some("v2"), false).is_ok());
        // v1 should be locked
        assert!(reg.check_delete("obj", Some("v1"), false).is_err());
    }
}
