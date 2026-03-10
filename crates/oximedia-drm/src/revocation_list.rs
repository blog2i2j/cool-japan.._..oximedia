//! Certificate and license revocation list management.
//!
//! Manages revocation lists for DRM certificates, device keys, and licenses.
//! Supports CRL (Certificate Revocation List) style operations with efficient
//! lookup and time-based expiry.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_arguments)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Reason for revocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RevocationReason {
    /// Key has been compromised.
    KeyCompromise,
    /// Certificate authority was compromised.
    CaCompromise,
    /// Affiliation of the entity changed.
    AffiliationChanged,
    /// Superseded by a newer certificate.
    Superseded,
    /// Entity has ceased operations.
    CessationOfOperation,
    /// Certificate is temporarily on hold.
    CertificateHold,
    /// Removed from CRL (un-revoked from hold).
    RemoveFromCrl,
    /// Privilege associated with certificate withdrawn.
    PrivilegeWithdrawn,
    /// License policy violation detected.
    PolicyViolation,
}

impl RevocationReason {
    /// Whether this reason is permanent (cannot be un-revoked).
    #[must_use]
    pub fn is_permanent(&self) -> bool {
        !matches!(self, Self::CertificateHold | Self::RemoveFromCrl)
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::KeyCompromise => "Key Compromise",
            Self::CaCompromise => "CA Compromise",
            Self::AffiliationChanged => "Affiliation Changed",
            Self::Superseded => "Superseded",
            Self::CessationOfOperation => "Cessation of Operation",
            Self::CertificateHold => "Certificate Hold",
            Self::RemoveFromCrl => "Remove from CRL",
            Self::PrivilegeWithdrawn => "Privilege Withdrawn",
            Self::PolicyViolation => "Policy Violation",
        }
    }
}

/// A single revocation entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevocationEntry {
    /// Identifier of the revoked entity (serial number, key ID, device ID, etc.).
    pub entity_id: String,
    /// Reason for revocation.
    pub reason: RevocationReason,
    /// Epoch timestamp (ms) when revocation was recorded.
    pub revoked_at_ms: u64,
    /// Optional epoch timestamp (ms) when revocation expires (for holds).
    pub expires_at_ms: Option<u64>,
    /// Who initiated the revocation.
    pub issuer: String,
    /// Optional note.
    pub note: Option<String>,
}

impl RevocationEntry {
    /// Create a new revocation entry.
    #[must_use]
    pub fn new(entity_id: &str, reason: RevocationReason, revoked_at_ms: u64, issuer: &str) -> Self {
        Self {
            entity_id: entity_id.to_string(),
            reason,
            revoked_at_ms,
            expires_at_ms: None,
            issuer: issuer.to_string(),
            note: None,
        }
    }

    /// Set an expiry time.
    #[must_use]
    pub fn with_expiry(mut self, expires_at_ms: u64) -> Self {
        self.expires_at_ms = Some(expires_at_ms);
        self
    }

    /// Set a note.
    #[must_use]
    pub fn with_note(mut self, note: &str) -> Self {
        self.note = Some(note.to_string());
        self
    }

    /// Whether this entry is expired at the given time.
    #[must_use]
    pub fn is_expired(&self, now_ms: u64) -> bool {
        self.expires_at_ms.map_or(false, |exp| now_ms >= exp)
    }

    /// Whether this entry is currently active (not expired) at the given time.
    #[must_use]
    pub fn is_active(&self, now_ms: u64) -> bool {
        !self.is_expired(now_ms)
    }
}

/// Summary statistics for a revocation list.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RevocationStats {
    /// Total entries.
    pub total: usize,
    /// Active (non-expired) entries.
    pub active: usize,
    /// Expired entries.
    pub expired: usize,
    /// Permanent revocations.
    pub permanent: usize,
    /// Entries on hold.
    pub on_hold: usize,
}

/// In-memory revocation list.
#[derive(Debug, Clone)]
pub struct RevocationList {
    /// Version / sequence number of this CRL.
    version: u64,
    /// Epoch timestamp (ms) when this list was last updated.
    updated_at_ms: u64,
    /// Entries keyed by entity ID.
    entries: HashMap<String, RevocationEntry>,
}

impl RevocationList {
    /// Create a new empty revocation list.
    #[must_use]
    pub fn new() -> Self {
        Self {
            version: 1,
            updated_at_ms: 0,
            entries: HashMap::new(),
        }
    }

    /// Current version of the list.
    #[must_use]
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Timestamp of last update.
    #[must_use]
    pub fn updated_at_ms(&self) -> u64 {
        self.updated_at_ms
    }

    /// Add a revocation entry.
    pub fn revoke(&mut self, entry: RevocationEntry) {
        self.updated_at_ms = entry.revoked_at_ms;
        self.version += 1;
        self.entries.insert(entry.entity_id.clone(), entry);
    }

    /// Remove an entry (un-revoke).  Returns the removed entry if present.
    pub fn unrevoke(&mut self, entity_id: &str) -> Option<RevocationEntry> {
        let removed = self.entries.remove(entity_id);
        if removed.is_some() {
            self.version += 1;
        }
        removed
    }

    /// Check whether an entity is revoked (active entry exists).
    #[must_use]
    pub fn is_revoked(&self, entity_id: &str, now_ms: u64) -> bool {
        self.entries
            .get(entity_id)
            .map_or(false, |e| e.is_active(now_ms))
    }

    /// Get the entry for an entity.
    #[must_use]
    pub fn get(&self, entity_id: &str) -> Option<&RevocationEntry> {
        self.entries.get(entity_id)
    }

    /// Number of entries in the list.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the list is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// All entity IDs currently in the list.
    #[must_use]
    pub fn entity_ids(&self) -> Vec<&str> {
        self.entries.keys().map(String::as_str).collect()
    }

    /// All entries matching a given reason.
    #[must_use]
    pub fn by_reason(&self, reason: RevocationReason) -> Vec<&RevocationEntry> {
        self.entries.values().filter(|e| e.reason == reason).collect()
    }

    /// Remove all expired entries.  Returns the number removed.
    pub fn purge_expired(&mut self, now_ms: u64) -> usize {
        let before = self.entries.len();
        self.entries.retain(|_, e| !e.is_expired(now_ms));
        let removed = before - self.entries.len();
        if removed > 0 {
            self.version += 1;
        }
        removed
    }

    /// Compute summary statistics at the given time.
    #[must_use]
    pub fn stats(&self, now_ms: u64) -> RevocationStats {
        let mut s = RevocationStats {
            total: self.entries.len(),
            ..Default::default()
        };
        for e in self.entries.values() {
            if e.is_expired(now_ms) {
                s.expired += 1;
            } else {
                s.active += 1;
            }
            if e.reason.is_permanent() {
                s.permanent += 1;
            }
            if e.reason == RevocationReason::CertificateHold {
                s.on_hold += 1;
            }
        }
        s
    }

    /// Merge another revocation list into this one.  Newer entries win.
    pub fn merge(&mut self, other: &RevocationList) {
        for (id, entry) in &other.entries {
            let dominated = self
                .entries
                .get(id)
                .map_or(true, |existing| entry.revoked_at_ms > existing.revoked_at_ms);
            if dominated {
                self.entries.insert(id.clone(), entry.clone());
            }
        }
        self.version += 1;
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.version += 1;
    }
}

impl Default for RevocationList {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_revocation_reason_permanent() {
        assert!(RevocationReason::KeyCompromise.is_permanent());
        assert!(RevocationReason::CaCompromise.is_permanent());
        assert!(RevocationReason::PolicyViolation.is_permanent());
        assert!(!RevocationReason::CertificateHold.is_permanent());
        assert!(!RevocationReason::RemoveFromCrl.is_permanent());
    }

    #[test]
    fn test_revocation_reason_label() {
        assert_eq!(RevocationReason::KeyCompromise.label(), "Key Compromise");
        assert_eq!(RevocationReason::Superseded.label(), "Superseded");
    }

    #[test]
    fn test_entry_new() {
        let e = RevocationEntry::new("dev-001", RevocationReason::KeyCompromise, 1000, "admin");
        assert_eq!(e.entity_id, "dev-001");
        assert_eq!(e.reason, RevocationReason::KeyCompromise);
        assert_eq!(e.revoked_at_ms, 1000);
        assert_eq!(e.issuer, "admin");
        assert!(e.expires_at_ms.is_none());
        assert!(e.note.is_none());
    }

    #[test]
    fn test_entry_builder() {
        let e = RevocationEntry::new("cert-42", RevocationReason::CertificateHold, 500, "sys")
            .with_expiry(2000)
            .with_note("temporary hold");
        assert_eq!(e.expires_at_ms, Some(2000));
        assert_eq!(e.note.as_deref(), Some("temporary hold"));
    }

    #[test]
    fn test_entry_expiry() {
        let e = RevocationEntry::new("x", RevocationReason::CertificateHold, 100, "a")
            .with_expiry(500);
        assert!(!e.is_expired(400));
        assert!(e.is_active(400));
        assert!(e.is_expired(500));
        assert!(!e.is_active(500));
        assert!(e.is_expired(600));
    }

    #[test]
    fn test_entry_no_expiry_never_expired() {
        let e = RevocationEntry::new("x", RevocationReason::KeyCompromise, 100, "a");
        assert!(!e.is_expired(999_999));
        assert!(e.is_active(999_999));
    }

    #[test]
    fn test_list_revoke_and_check() {
        let mut list = RevocationList::new();
        let entry = RevocationEntry::new("dev-1", RevocationReason::PolicyViolation, 1000, "admin");
        list.revoke(entry);
        assert!(list.is_revoked("dev-1", 1500));
        assert!(!list.is_revoked("dev-2", 1500));
    }

    #[test]
    fn test_list_unrevoke() {
        let mut list = RevocationList::new();
        list.revoke(RevocationEntry::new("a", RevocationReason::CertificateHold, 100, "x"));
        assert!(list.is_revoked("a", 200));
        let removed = list.unrevoke("a");
        assert!(removed.is_some());
        assert!(!list.is_revoked("a", 200));
        // Un-revoking a non-existent entry returns None.
        assert!(list.unrevoke("zzz").is_none());
    }

    #[test]
    fn test_list_purge_expired() {
        let mut list = RevocationList::new();
        list.revoke(
            RevocationEntry::new("a", RevocationReason::CertificateHold, 100, "x").with_expiry(500),
        );
        list.revoke(RevocationEntry::new("b", RevocationReason::KeyCompromise, 200, "x"));
        assert_eq!(list.len(), 2);
        let purged = list.purge_expired(600);
        assert_eq!(purged, 1);
        assert_eq!(list.len(), 1);
        assert!(list.get("b").is_some());
    }

    #[test]
    fn test_list_by_reason() {
        let mut list = RevocationList::new();
        list.revoke(RevocationEntry::new("a", RevocationReason::KeyCompromise, 100, "x"));
        list.revoke(RevocationEntry::new("b", RevocationReason::Superseded, 200, "x"));
        list.revoke(RevocationEntry::new("c", RevocationReason::KeyCompromise, 300, "x"));
        let compromised = list.by_reason(RevocationReason::KeyCompromise);
        assert_eq!(compromised.len(), 2);
    }

    #[test]
    fn test_list_stats() {
        let mut list = RevocationList::new();
        list.revoke(RevocationEntry::new("a", RevocationReason::KeyCompromise, 100, "x"));
        list.revoke(
            RevocationEntry::new("b", RevocationReason::CertificateHold, 200, "x").with_expiry(500),
        );
        list.revoke(
            RevocationEntry::new("c", RevocationReason::Superseded, 300, "x").with_expiry(400),
        );
        let stats = list.stats(450);
        assert_eq!(stats.total, 3);
        assert_eq!(stats.active, 2); // a (no expiry), b (expires at 500 > 450)
        assert_eq!(stats.expired, 1); // c (expires at 400 < 450)
        assert_eq!(stats.permanent, 2); // a, c
        assert_eq!(stats.on_hold, 1); // b
    }

    #[test]
    fn test_list_merge() {
        let mut list1 = RevocationList::new();
        list1.revoke(RevocationEntry::new("a", RevocationReason::KeyCompromise, 100, "x"));
        list1.revoke(RevocationEntry::new("b", RevocationReason::Superseded, 200, "x"));

        let mut list2 = RevocationList::new();
        // Newer entry for "a" wins.
        list2.revoke(RevocationEntry::new("a", RevocationReason::PolicyViolation, 300, "y"));
        list2.revoke(RevocationEntry::new("c", RevocationReason::CaCompromise, 400, "y"));

        list1.merge(&list2);
        assert_eq!(list1.len(), 3);
        assert_eq!(list1.get("a").expect("entry should exist").reason, RevocationReason::PolicyViolation);
        assert!(list1.get("c").is_some());
    }

    #[test]
    fn test_list_version_increments() {
        let mut list = RevocationList::new();
        assert_eq!(list.version(), 1);
        list.revoke(RevocationEntry::new("a", RevocationReason::KeyCompromise, 100, "x"));
        assert_eq!(list.version(), 2);
        list.unrevoke("a");
        assert_eq!(list.version(), 3);
    }

    #[test]
    fn test_list_clear() {
        let mut list = RevocationList::new();
        list.revoke(RevocationEntry::new("a", RevocationReason::KeyCompromise, 100, "x"));
        list.revoke(RevocationEntry::new("b", RevocationReason::Superseded, 200, "x"));
        assert!(!list.is_empty());
        list.clear();
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn test_list_entity_ids() {
        let mut list = RevocationList::new();
        list.revoke(RevocationEntry::new("alpha", RevocationReason::KeyCompromise, 100, "x"));
        list.revoke(RevocationEntry::new("beta", RevocationReason::Superseded, 200, "x"));
        let mut ids = list.entity_ids();
        ids.sort();
        assert_eq!(ids, vec!["alpha", "beta"]);
    }

    #[test]
    fn test_default_impl() {
        let list = RevocationList::default();
        assert!(list.is_empty());
        assert_eq!(list.version(), 1);
    }
}
