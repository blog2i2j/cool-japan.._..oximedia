#![allow(dead_code)]
//! In-memory rights record database for fast lookup and management.

use std::collections::HashMap;

/// The status of a rights record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordStatus {
    /// The right is currently active and enforceable.
    Active,
    /// The right has expired.
    Expired,
    /// The right has been revoked before its natural expiry.
    Revoked,
    /// The record is pending verification.
    Pending,
}

/// A single rights record stored in the database.
#[derive(Debug, Clone)]
pub struct RightsRecord {
    /// Unique identifier for this record.
    pub id: String,
    /// The asset this record applies to.
    pub asset_id: String,
    /// The rights holder (person or organisation).
    pub holder: String,
    /// Current status of this record.
    pub status: RecordStatus,
    /// Unix timestamp when the right was granted.
    pub granted_at: u64,
    /// Unix timestamp when the right expires, if any.
    pub expires_at: Option<u64>,
}

impl RightsRecord {
    /// Create a new `RightsRecord`.
    pub fn new(
        id: impl Into<String>,
        asset_id: impl Into<String>,
        holder: impl Into<String>,
        status: RecordStatus,
        granted_at: u64,
        expires_at: Option<u64>,
    ) -> Self {
        Self {
            id: id.into(),
            asset_id: asset_id.into(),
            holder: holder.into(),
            status,
            granted_at,
            expires_at,
        }
    }

    /// Return `true` if this record is currently active.
    pub fn is_active(&self) -> bool {
        self.status == RecordStatus::Active
    }
}

/// An in-memory store of [`RightsRecord`] entries, indexed by record ID.
#[derive(Debug, Default)]
pub struct RightsDatabase {
    records: HashMap<String, RightsRecord>,
}

impl RightsDatabase {
    /// Create a new, empty `RightsDatabase`.
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
        }
    }

    /// Insert a record into the database.
    ///
    /// If a record with the same `id` already exists it is replaced.
    pub fn insert(&mut self, record: RightsRecord) {
        self.records.insert(record.id.clone(), record);
    }

    /// Look up a record by its unique ID.
    ///
    /// Returns `None` if no record with that ID exists.
    pub fn lookup(&self, id: &str) -> Option<&RightsRecord> {
        self.records.get(id)
    }

    /// Update the status of an existing record.
    ///
    /// Returns `true` if the record was found and updated, `false` otherwise.
    pub fn update_status(&mut self, id: &str, new_status: RecordStatus) -> bool {
        if let Some(record) = self.records.get_mut(id) {
            record.status = new_status;
            true
        } else {
            false
        }
    }

    /// Return references to all records whose status is [`RecordStatus::Active`].
    pub fn active_records(&self) -> Vec<&RightsRecord> {
        self.records.values().filter(|r| r.is_active()).collect()
    }

    /// Return the total number of records in the database.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Return `true` if the database contains no records.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Remove a record by ID, returning it if it existed.
    pub fn remove(&mut self, id: &str) -> Option<RightsRecord> {
        self.records.remove(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(id: &str, status: RecordStatus) -> RightsRecord {
        RightsRecord::new(id, "asset-1", "Alice", status, 1000, Some(9999))
    }

    #[test]
    fn test_record_is_active_true() {
        let r = make_record("r1", RecordStatus::Active);
        assert!(r.is_active());
    }

    #[test]
    fn test_record_is_active_false_expired() {
        let r = make_record("r2", RecordStatus::Expired);
        assert!(!r.is_active());
    }

    #[test]
    fn test_record_is_active_false_revoked() {
        let r = make_record("r3", RecordStatus::Revoked);
        assert!(!r.is_active());
    }

    #[test]
    fn test_record_is_active_false_pending() {
        let r = make_record("r4", RecordStatus::Pending);
        assert!(!r.is_active());
    }

    #[test]
    fn test_database_initially_empty() {
        let db = RightsDatabase::new();
        assert!(db.is_empty());
        assert_eq!(db.len(), 0);
    }

    #[test]
    fn test_insert_and_lookup() {
        let mut db = RightsDatabase::new();
        db.insert(make_record("r1", RecordStatus::Active));
        let found = db.lookup("r1");
        assert!(found.is_some());
        assert_eq!(
            found.expect("rights test operation should succeed").id,
            "r1"
        );
    }

    #[test]
    fn test_lookup_missing_returns_none() {
        let db = RightsDatabase::new();
        assert!(db.lookup("nonexistent").is_none());
    }

    #[test]
    fn test_insert_replaces_existing() {
        let mut db = RightsDatabase::new();
        db.insert(make_record("r1", RecordStatus::Active));
        db.insert(make_record("r1", RecordStatus::Revoked));
        let r = db
            .lookup("r1")
            .expect("rights test operation should succeed");
        assert_eq!(r.status, RecordStatus::Revoked);
    }

    #[test]
    fn test_update_status_success() {
        let mut db = RightsDatabase::new();
        db.insert(make_record("r1", RecordStatus::Active));
        let updated = db.update_status("r1", RecordStatus::Expired);
        assert!(updated);
        assert_eq!(
            db.lookup("r1")
                .expect("rights test operation should succeed")
                .status,
            RecordStatus::Expired
        );
    }

    #[test]
    fn test_update_status_missing_returns_false() {
        let mut db = RightsDatabase::new();
        assert!(!db.update_status("ghost", RecordStatus::Revoked));
    }

    #[test]
    fn test_active_records_filters_correctly() {
        let mut db = RightsDatabase::new();
        db.insert(make_record("r1", RecordStatus::Active));
        db.insert(make_record("r2", RecordStatus::Expired));
        db.insert(make_record("r3", RecordStatus::Active));
        let active = db.active_records();
        assert_eq!(active.len(), 2);
    }

    #[test]
    fn test_remove_existing() {
        let mut db = RightsDatabase::new();
        db.insert(make_record("r1", RecordStatus::Active));
        let removed = db.remove("r1");
        assert!(removed.is_some());
        assert!(db.is_empty());
    }

    #[test]
    fn test_remove_missing_returns_none() {
        let mut db = RightsDatabase::new();
        assert!(db.remove("ghost").is_none());
    }

    #[test]
    fn test_len_after_multiple_inserts() {
        let mut db = RightsDatabase::new();
        db.insert(make_record("r1", RecordStatus::Active));
        db.insert(make_record("r2", RecordStatus::Pending));
        db.insert(make_record("r3", RecordStatus::Revoked));
        assert_eq!(db.len(), 3);
    }
}
