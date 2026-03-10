//! Rights clearance management system

#![allow(dead_code)]

/// Status of a clearance request
#[derive(Debug, Clone, PartialEq)]
pub enum ClearanceStatus {
    /// Request submitted, awaiting review
    Pending,
    /// Clearance review is underway
    InProgress,
    /// Clearance granted
    Cleared,
    /// Clearance denied
    Rejected {
        /// Reason for rejection
        reason: String,
    },
    /// Clearance previously granted has expired
    Expired,
}

impl ClearanceStatus {
    /// Returns true if this status represents a terminal (resolved) state
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            ClearanceStatus::Cleared | ClearanceStatus::Rejected { .. } | ClearanceStatus::Expired
        )
    }
}

/// A request to obtain rights clearance for a piece of content
#[derive(Debug, Clone)]
pub struct ClearanceRequest {
    /// Unique identifier for this request
    pub id: u64,
    /// Identifier for the content requiring clearance
    pub content_id: String,
    /// Type of right being cleared (e.g. "Music Sync", "Footage", "Talent")
    pub right_type: String,
    /// Entity or person submitting the request
    pub requestor: String,
    /// Unix timestamp when the request was submitted
    pub submitted_at: u64,
    /// Territory for which clearance is requested
    pub territory: String,
}

impl ClearanceRequest {
    /// Create a new clearance request
    pub fn new(
        id: u64,
        content_id: impl Into<String>,
        right_type: impl Into<String>,
        requestor: impl Into<String>,
        submitted_at: u64,
        territory: impl Into<String>,
    ) -> Self {
        Self {
            id,
            content_id: content_id.into(),
            right_type: right_type.into(),
            requestor: requestor.into(),
            submitted_at,
            territory: territory.into(),
        }
    }
}

/// A full clearance record combining request and its current status
#[derive(Debug, Clone)]
pub struct ClearanceRecord {
    /// The original clearance request
    pub request: ClearanceRequest,
    /// Current status of the clearance
    pub status: ClearanceStatus,
    /// Unix timestamp when the request was resolved (if applicable)
    pub resolved_at: Option<u64>,
    /// Additional notes from the reviewer
    pub notes: String,
}

impl ClearanceRecord {
    /// Create a new clearance record for a submitted request (status: Pending)
    pub fn new(request: ClearanceRequest) -> Self {
        Self {
            request,
            status: ClearanceStatus::Pending,
            resolved_at: None,
            notes: String::new(),
        }
    }

    /// Approve the clearance request
    pub fn approve(&mut self, now: u64, notes: &str) {
        self.status = ClearanceStatus::Cleared;
        self.resolved_at = Some(now);
        self.notes = notes.to_string();
    }

    /// Reject the clearance request with a reason
    pub fn reject(&mut self, now: u64, reason: &str) {
        self.status = ClearanceStatus::Rejected {
            reason: reason.to_string(),
        };
        self.resolved_at = Some(now);
        self.notes = reason.to_string();
    }

    /// Mark the clearance as in-progress (review started)
    pub fn start_review(&mut self) {
        if self.status == ClearanceStatus::Pending {
            self.status = ClearanceStatus::InProgress;
        }
    }

    /// Mark a previously cleared request as expired
    pub fn expire(&mut self, now: u64) {
        if self.status == ClearanceStatus::Cleared {
            self.status = ClearanceStatus::Expired;
            self.resolved_at = Some(now);
        }
    }

    /// Returns true if the request has been resolved (cleared, rejected, or expired)
    pub fn is_resolved(&self) -> bool {
        self.status.is_terminal()
    }

    /// Number of full days the request has been pending (or was pending until resolution)
    pub fn days_pending(&self, now: u64) -> u64 {
        let end = self.resolved_at.unwrap_or(now);
        let elapsed_secs = end.saturating_sub(self.request.submitted_at);
        elapsed_secs / 86_400
    }
}

/// An in-memory database of clearance records
#[derive(Debug, Default)]
pub struct ClearanceDatabase {
    records: Vec<ClearanceRecord>,
}

impl ClearanceDatabase {
    /// Create an empty clearance database
    pub fn new() -> Self {
        Self::default()
    }

    /// Submit a new clearance request and return its ID
    pub fn submit(&mut self, request: ClearanceRequest) -> u64 {
        let id = request.id;
        self.records.push(ClearanceRecord::new(request));
        id
    }

    /// Look up a record by ID
    pub fn get(&self, id: u64) -> Option<&ClearanceRecord> {
        self.records.iter().find(|r| r.request.id == id)
    }

    /// Look up a mutable record by ID
    pub fn get_mut(&mut self, id: u64) -> Option<&mut ClearanceRecord> {
        self.records.iter_mut().find(|r| r.request.id == id)
    }

    /// Count requests that are still pending (Pending or InProgress)
    pub fn pending_count(&self) -> usize {
        self.records
            .iter()
            .filter(|r| {
                r.status == ClearanceStatus::Pending || r.status == ClearanceStatus::InProgress
            })
            .count()
    }

    /// Return content_ids whose clearance status is Cleared
    pub fn cleared_content(&self) -> Vec<&str> {
        self.records
            .iter()
            .filter(|r| r.status == ClearanceStatus::Cleared)
            .map(|r| r.request.content_id.as_str())
            .collect()
    }

    /// Total number of records in the database
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns true if no records have been submitted
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_request(id: u64) -> ClearanceRequest {
        ClearanceRequest::new(
            id,
            format!("content-{id}"),
            "Music Sync",
            "Alice",
            1_000_000,
            "US",
        )
    }

    #[test]
    fn test_new_record_is_pending() {
        let rec = ClearanceRecord::new(make_request(1));
        assert_eq!(rec.status, ClearanceStatus::Pending);
        assert!(rec.resolved_at.is_none());
        assert!(!rec.is_resolved());
    }

    #[test]
    fn test_approve_sets_cleared() {
        let mut rec = ClearanceRecord::new(make_request(2));
        rec.approve(2_000_000, "Looks good");
        assert_eq!(rec.status, ClearanceStatus::Cleared);
        assert_eq!(rec.resolved_at, Some(2_000_000));
        assert!(rec.is_resolved());
    }

    #[test]
    fn test_reject_sets_rejected() {
        let mut rec = ClearanceRecord::new(make_request(3));
        rec.reject(2_000_000, "Rights not available");
        assert!(matches!(
            rec.status,
            ClearanceStatus::Rejected { ref reason } if reason == "Rights not available"
        ));
        assert!(rec.is_resolved());
    }

    #[test]
    fn test_start_review_changes_pending_to_in_progress() {
        let mut rec = ClearanceRecord::new(make_request(4));
        rec.start_review();
        assert_eq!(rec.status, ClearanceStatus::InProgress);
    }

    #[test]
    fn test_start_review_does_not_change_resolved() {
        let mut rec = ClearanceRecord::new(make_request(5));
        rec.approve(2_000_000, "ok");
        rec.start_review();
        assert_eq!(rec.status, ClearanceStatus::Cleared);
    }

    #[test]
    fn test_expire_cleared_record() {
        let mut rec = ClearanceRecord::new(make_request(6));
        rec.approve(2_000_000, "ok");
        rec.expire(3_000_000);
        assert_eq!(rec.status, ClearanceStatus::Expired);
        assert_eq!(rec.resolved_at, Some(3_000_000));
    }

    #[test]
    fn test_days_pending_no_resolution() {
        let req = ClearanceRequest::new(7, "c", "r", "r", 0, "US");
        let rec = ClearanceRecord::new(req);
        // 2 days of seconds
        let days = rec.days_pending(2 * 86_400);
        assert_eq!(days, 2);
    }

    #[test]
    fn test_days_pending_resolved() {
        let req = ClearanceRequest::new(8, "c", "r", "r", 0, "US");
        let mut rec = ClearanceRecord::new(req);
        rec.approve(5 * 86_400, "ok");
        // resolved_at used, not `now`
        let days = rec.days_pending(100 * 86_400);
        assert_eq!(days, 5);
    }

    #[test]
    fn test_database_submit_and_get() {
        let mut db = ClearanceDatabase::new();
        let id = db.submit(make_request(10));
        assert_eq!(id, 10);
        assert!(db.get(10).is_some());
    }

    #[test]
    fn test_database_pending_count() {
        let mut db = ClearanceDatabase::new();
        db.submit(make_request(11));
        db.submit(make_request(12));
        let rec = db
            .get_mut(11)
            .expect("rights test operation should succeed");
        rec.approve(999, "ok");
        assert_eq!(db.pending_count(), 1);
    }

    #[test]
    fn test_database_cleared_content() {
        let mut db = ClearanceDatabase::new();
        db.submit(make_request(13));
        db.submit(make_request(14));
        let rec = db
            .get_mut(13)
            .expect("rights test operation should succeed");
        rec.approve(999, "ok");
        let cleared = db.cleared_content();
        assert_eq!(cleared.len(), 1);
        assert_eq!(cleared[0], "content-13");
    }

    #[test]
    fn test_database_get_unknown_id_returns_none() {
        let db = ClearanceDatabase::new();
        assert!(db.get(999).is_none());
    }

    #[test]
    fn test_database_is_empty() {
        let db = ClearanceDatabase::new();
        assert!(db.is_empty());
        assert_eq!(db.len(), 0);
    }
}
