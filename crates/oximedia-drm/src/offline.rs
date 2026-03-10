//! Offline DRM license management.
//!
//! Provides persistent offline license storage, expiry checking, and renewal support.

use serde::{Deserialize, Serialize};

/// An offline DRM license stored locally on a device
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct OfflineLicense {
    pub key_id: Vec<u8>,
    pub encrypted_key: Vec<u8>,
    /// Expiry time as milliseconds since Unix epoch
    pub expiry_ms: u64,
    pub renewable: bool,
    pub renewal_url: Option<String>,
}

impl OfflineLicense {
    /// Create a new offline license
    pub fn new(key_id: Vec<u8>, encrypted_key: Vec<u8>, expiry_ms: u64) -> Self {
        Self {
            key_id,
            encrypted_key,
            expiry_ms,
            renewable: false,
            renewal_url: None,
        }
    }

    /// Mark the license as renewable with a renewal endpoint
    pub fn with_renewal(mut self, url: impl Into<String>) -> Self {
        self.renewable = true;
        self.renewal_url = Some(url.into());
        self
    }

    /// True if the license has expired at the given timestamp (ms)
    pub fn is_expired(&self, current_ms: u64) -> bool {
        current_ms >= self.expiry_ms
    }
}

/// Storage for offline licenses
pub struct OfflineLicenseStore {
    licenses: Vec<OfflineLicense>,
}

impl OfflineLicenseStore {
    /// Create an empty store
    pub fn new() -> Self {
        Self {
            licenses: Vec::new(),
        }
    }

    /// Persist a new or updated license
    pub fn persist(&mut self, license: OfflineLicense) {
        // Replace existing entry with the same key_id if present
        if let Some(existing) = self
            .licenses
            .iter_mut()
            .find(|l| l.key_id == license.key_id)
        {
            *existing = license;
        } else {
            self.licenses.push(license);
        }
    }

    /// Retrieve a license by key_id
    pub fn retrieve(&self, key_id: &[u8]) -> Option<&OfflineLicense> {
        self.licenses.iter().find(|l| l.key_id == key_id)
    }

    /// Remove all licenses that have expired at `current_ms`.
    ///
    /// Returns the number of licenses purged.
    pub fn purge_expired(&mut self, current_ms: u64) -> usize {
        let before = self.licenses.len();
        self.licenses.retain(|l| !l.is_expired(current_ms));
        before - self.licenses.len()
    }

    /// Number of stored licenses
    pub fn len(&self) -> usize {
        self.licenses.len()
    }

    /// True when no licenses are stored
    pub fn is_empty(&self) -> bool {
        self.licenses.is_empty()
    }
}

impl Default for OfflineLicenseStore {
    fn default() -> Self {
        Self::new()
    }
}

/// A request to renew an expiring or expired license
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct RenewalRequest {
    pub key_id: Vec<u8>,
    pub current_expiry_ms: u64,
    pub client_token: String,
}

impl RenewalRequest {
    /// Create a new renewal request
    pub fn new(key_id: Vec<u8>, current_expiry_ms: u64, client_token: impl Into<String>) -> Self {
        Self {
            key_id,
            current_expiry_ms,
            client_token: client_token.into(),
        }
    }
}

/// Response to a renewal request
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct RenewalResponse {
    pub new_expiry_ms: u64,
    pub success: bool,
    pub error: Option<String>,
}

impl RenewalResponse {
    /// Successful renewal
    pub fn ok(new_expiry_ms: u64) -> Self {
        Self {
            new_expiry_ms,
            success: true,
            error: None,
        }
    }

    /// Failed renewal
    pub fn err(msg: impl Into<String>) -> Self {
        Self {
            new_expiry_ms: 0,
            success: false,
            error: Some(msg.into()),
        }
    }
}

/// Status of an offline license
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum LicenseStatus {
    /// License is valid and not close to expiry
    Valid,
    /// License is valid but expires in `u64` milliseconds
    Expiring(u64),
    /// License has expired
    Expired,
    /// No license found for the key_id
    NotFound,
}

impl LicenseStatus {
    /// True when the license needs renewal (expiring soon or already expired)
    pub fn needs_renewal(&self) -> bool {
        matches!(self, LicenseStatus::Expiring(_) | LicenseStatus::Expired)
    }
}

/// Manager that checks offline license status and coordinates renewal
pub struct OfflineLicenseManager {
    store: OfflineLicenseStore,
    /// Milliseconds before expiry at which a license is considered "Expiring"
    renewal_threshold_ms: u64,
}

impl OfflineLicenseManager {
    /// Create a new manager with the given store and renewal threshold
    pub fn new(store: OfflineLicenseStore, renewal_threshold_ms: u64) -> Self {
        Self {
            store,
            renewal_threshold_ms,
        }
    }

    /// Check the expiry status of a license identified by `key_id`
    pub fn check_expiry(&self, key_id: &[u8], current_ms: u64) -> LicenseStatus {
        match self.store.retrieve(key_id) {
            None => LicenseStatus::NotFound,
            Some(lic) if lic.is_expired(current_ms) => LicenseStatus::Expired,
            Some(lic) => {
                let remaining_ms = lic.expiry_ms.saturating_sub(current_ms);
                if remaining_ms <= self.renewal_threshold_ms {
                    LicenseStatus::Expiring(remaining_ms)
                } else {
                    LicenseStatus::Valid
                }
            }
        }
    }

    /// Simulate a renewal: extends the license expiry by `extend_ms` milliseconds.
    ///
    /// Returns an error if the license is not found or not renewable.
    pub fn renew(&mut self, request: &RenewalRequest, extend_ms: u64) -> RenewalResponse {
        let new_expiry = match self.store.retrieve(&request.key_id) {
            None => return RenewalResponse::err("License not found"),
            Some(lic) if !lic.renewable => return RenewalResponse::err("License is not renewable"),
            Some(lic) => {
                // Extend from either current expiry or now (whichever is later)
                let base = lic.expiry_ms.max(request.current_expiry_ms);
                base + extend_ms
            }
        };

        // Persist the updated license
        if let Some(lic) = self.store.retrieve(&request.key_id) {
            let mut updated = lic.clone();
            updated.expiry_ms = new_expiry;
            self.store.persist(updated);
        }

        RenewalResponse::ok(new_expiry)
    }

    /// Delegate to store: purge all expired licenses
    pub fn purge_expired(&mut self, current_ms: u64) -> usize {
        self.store.purge_expired(current_ms)
    }

    /// Mutable access to the underlying store
    pub fn store_mut(&mut self) -> &mut OfflineLicenseStore {
        &mut self.store
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_license(key_id: u8, expiry_ms: u64) -> OfflineLicense {
        OfflineLicense::new(vec![key_id], vec![0xABu8; 16], expiry_ms)
    }

    #[test]
    fn test_persist_and_retrieve() {
        let mut store = OfflineLicenseStore::new();
        let lic = make_license(1, 10_000);
        store.persist(lic.clone());
        assert!(store.retrieve(&[1]).is_some());
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_retrieve_not_found() {
        let store = OfflineLicenseStore::new();
        assert!(store.retrieve(&[99]).is_none());
    }

    #[test]
    fn test_persist_updates_existing() {
        let mut store = OfflineLicenseStore::new();
        store.persist(make_license(1, 5_000));
        store.persist(make_license(1, 20_000));
        assert_eq!(store.len(), 1);
        assert_eq!(
            store
                .retrieve(&[1])
                .expect("license should exist")
                .expiry_ms,
            20_000
        );
    }

    #[test]
    fn test_purge_expired() {
        let mut store = OfflineLicenseStore::new();
        store.persist(make_license(1, 1_000)); // expired at t=2000
        store.persist(make_license(2, 5_000)); // still valid
        let purged = store.purge_expired(2_000);
        assert_eq!(purged, 1);
        assert_eq!(store.len(), 1);
        assert!(store.retrieve(&[2]).is_some());
    }

    #[test]
    fn test_is_expired() {
        let lic = make_license(1, 1_000);
        assert!(!lic.is_expired(999));
        assert!(lic.is_expired(1_000));
        assert!(lic.is_expired(1_001));
    }

    #[test]
    fn test_license_status_not_found() {
        let store = OfflineLicenseStore::new();
        let mgr = OfflineLicenseManager::new(store, 60_000);
        assert_eq!(mgr.check_expiry(&[1], 0), LicenseStatus::NotFound);
    }

    #[test]
    fn test_license_status_valid() {
        let mut store = OfflineLicenseStore::new();
        store.persist(make_license(1, 100_000));
        let mgr = OfflineLicenseManager::new(store, 5_000);
        // 100 000 ms expiry, 5 000 threshold, current = 0 → 100 000 ms remaining > 5 000
        assert_eq!(mgr.check_expiry(&[1], 0), LicenseStatus::Valid);
    }

    #[test]
    fn test_license_status_expiring() {
        let mut store = OfflineLicenseStore::new();
        store.persist(make_license(1, 10_000));
        let mgr = OfflineLicenseManager::new(store, 5_000);
        // remaining = 10 000 - 6 000 = 4 000 ≤ 5 000 → Expiring
        assert_eq!(
            mgr.check_expiry(&[1], 6_000),
            LicenseStatus::Expiring(4_000)
        );
    }

    #[test]
    fn test_license_status_expired() {
        let mut store = OfflineLicenseStore::new();
        store.persist(make_license(1, 1_000));
        let mgr = OfflineLicenseManager::new(store, 5_000);
        assert_eq!(mgr.check_expiry(&[1], 2_000), LicenseStatus::Expired);
    }

    #[test]
    fn test_needs_renewal() {
        assert!(LicenseStatus::Expiring(100).needs_renewal());
        assert!(LicenseStatus::Expired.needs_renewal());
        assert!(!LicenseStatus::Valid.needs_renewal());
        assert!(!LicenseStatus::NotFound.needs_renewal());
    }

    #[test]
    fn test_renewal_success() {
        let mut store = OfflineLicenseStore::new();
        store.persist(
            OfflineLicense::new(vec![1], vec![0u8; 16], 5_000)
                .with_renewal("https://example.com/renew"),
        );
        let mut mgr = OfflineLicenseManager::new(store, 1_000);
        let req = RenewalRequest::new(vec![1], 5_000, "token-abc");
        let resp = mgr.renew(&req, 86_400_000);
        assert!(resp.success);
        assert_eq!(resp.new_expiry_ms, 5_000 + 86_400_000);
    }

    #[test]
    fn test_renewal_not_found() {
        let store = OfflineLicenseStore::new();
        let mut mgr = OfflineLicenseManager::new(store, 1_000);
        let req = RenewalRequest::new(vec![99], 0, "tok");
        let resp = mgr.renew(&req, 1_000);
        assert!(!resp.success);
        assert!(resp.error.is_some());
    }

    #[test]
    fn test_renewal_not_renewable() {
        let mut store = OfflineLicenseStore::new();
        store.persist(make_license(2, 5_000)); // renewable = false
        let mut mgr = OfflineLicenseManager::new(store, 1_000);
        let req = RenewalRequest::new(vec![2], 5_000, "tok");
        let resp = mgr.renew(&req, 1_000);
        assert!(!resp.success);
    }
}
