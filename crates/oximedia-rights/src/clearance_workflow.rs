#![allow(dead_code)]
//! Clearance workflow management for media rights approvals.

/// The status of a clearance request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClearanceStatus {
    /// The request is awaiting review.
    Pending,
    /// The clearance has been approved.
    Approved,
    /// The clearance has been rejected.
    Rejected,
    /// More information is required before a decision can be made.
    NeedsInfo,
    /// The request has been escalated to a senior reviewer.
    Escalated,
}

impl ClearanceStatus {
    /// Return `true` if this status blocks use of the asset.
    ///
    /// A status is blocking when it is not yet `Approved`.
    pub fn is_blocking(&self) -> bool {
        !matches!(self, ClearanceStatus::Approved)
    }
}

/// A request to clear the rights for a specific asset and usage context.
#[derive(Debug, Clone)]
pub struct ClearanceRequest {
    /// Unique identifier for this request.
    pub id: String,
    /// The asset requiring clearance.
    pub asset_id: String,
    /// A description of the intended usage.
    pub usage_description: String,
    /// Current status of this request.
    pub status: ClearanceStatus,
    /// Unix timestamp when the request was submitted.
    pub submitted_at: u64,
    /// Whether this request is marked as urgent.
    pub urgent: bool,
}

impl ClearanceRequest {
    /// Create a new `ClearanceRequest`.
    pub fn new(
        id: impl Into<String>,
        asset_id: impl Into<String>,
        usage_description: impl Into<String>,
        submitted_at: u64,
        urgent: bool,
    ) -> Self {
        Self {
            id: id.into(),
            asset_id: asset_id.into(),
            usage_description: usage_description.into(),
            status: ClearanceStatus::Pending,
            submitted_at,
            urgent,
        }
    }

    /// Return `true` if this request is marked as urgent.
    pub fn is_urgent(&self) -> bool {
        self.urgent
    }
}

/// A workflow engine that tracks and manages clearance requests.
#[derive(Debug, Default)]
pub struct ClearanceWorkflow {
    requests: Vec<ClearanceRequest>,
}

impl ClearanceWorkflow {
    /// Create a new, empty `ClearanceWorkflow`.
    pub fn new() -> Self {
        Self {
            requests: Vec::new(),
        }
    }

    /// Submit a new clearance request to the workflow.
    ///
    /// The request is stored with status [`ClearanceStatus::Pending`].
    pub fn submit(&mut self, request: ClearanceRequest) {
        self.requests.push(request);
    }

    /// Approve the clearance request with the given ID.
    ///
    /// Returns `true` if the request was found and its status updated to
    /// [`ClearanceStatus::Approved`], `false` if no request with that ID exists.
    pub fn approve(&mut self, id: &str) -> bool {
        self.set_status(id, ClearanceStatus::Approved)
    }

    /// Reject the clearance request with the given ID.
    ///
    /// Returns `true` if the request was found and updated, `false` otherwise.
    pub fn reject(&mut self, id: &str) -> bool {
        self.set_status(id, ClearanceStatus::Rejected)
    }

    /// Escalate the clearance request with the given ID.
    ///
    /// Returns `true` if the request was found and updated, `false` otherwise.
    pub fn escalate(&mut self, id: &str) -> bool {
        self.set_status(id, ClearanceStatus::Escalated)
    }

    /// Return the number of requests currently in `Pending` status.
    pub fn pending_count(&self) -> usize {
        self.requests
            .iter()
            .filter(|r| r.status == ClearanceStatus::Pending)
            .count()
    }

    /// Return references to all urgent pending requests.
    pub fn urgent_pending(&self) -> Vec<&ClearanceRequest> {
        self.requests
            .iter()
            .filter(|r| r.status == ClearanceStatus::Pending && r.urgent)
            .collect()
    }

    /// Look up a request by its ID.
    ///
    /// Returns `None` if no request with that ID exists.
    pub fn lookup(&self, id: &str) -> Option<&ClearanceRequest> {
        self.requests.iter().find(|r| r.id == id)
    }

    /// Return the total number of requests in the workflow.
    pub fn total_count(&self) -> usize {
        self.requests.len()
    }

    // Internal helper to update status by ID.
    fn set_status(&mut self, id: &str, new_status: ClearanceStatus) -> bool {
        if let Some(req) = self.requests.iter_mut().find(|r| r.id == id) {
            req.status = new_status;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_req(id: &str, urgent: bool) -> ClearanceRequest {
        ClearanceRequest::new(id, "asset-1", "broadcast use", 1000, urgent)
    }

    #[test]
    fn test_clearance_status_pending_is_blocking() {
        assert!(ClearanceStatus::Pending.is_blocking());
    }

    #[test]
    fn test_clearance_status_rejected_is_blocking() {
        assert!(ClearanceStatus::Rejected.is_blocking());
    }

    #[test]
    fn test_clearance_status_needs_info_is_blocking() {
        assert!(ClearanceStatus::NeedsInfo.is_blocking());
    }

    #[test]
    fn test_clearance_status_escalated_is_blocking() {
        assert!(ClearanceStatus::Escalated.is_blocking());
    }

    #[test]
    fn test_clearance_status_approved_not_blocking() {
        assert!(!ClearanceStatus::Approved.is_blocking());
    }

    #[test]
    fn test_request_is_urgent() {
        let req = make_req("r1", true);
        assert!(req.is_urgent());
    }

    #[test]
    fn test_request_not_urgent() {
        let req = make_req("r1", false);
        assert!(!req.is_urgent());
    }

    #[test]
    fn test_request_initial_status_is_pending() {
        let req = make_req("r1", false);
        assert_eq!(req.status, ClearanceStatus::Pending);
    }

    #[test]
    fn test_workflow_submit_increases_count() {
        let mut wf = ClearanceWorkflow::new();
        wf.submit(make_req("r1", false));
        wf.submit(make_req("r2", false));
        assert_eq!(wf.total_count(), 2);
        assert_eq!(wf.pending_count(), 2);
    }

    #[test]
    fn test_approve_changes_status() {
        let mut wf = ClearanceWorkflow::new();
        wf.submit(make_req("r1", false));
        assert!(wf.approve("r1"));
        let req = wf
            .lookup("r1")
            .expect("rights test operation should succeed");
        assert_eq!(req.status, ClearanceStatus::Approved);
    }

    #[test]
    fn test_approve_decreases_pending_count() {
        let mut wf = ClearanceWorkflow::new();
        wf.submit(make_req("r1", false));
        wf.submit(make_req("r2", false));
        wf.approve("r1");
        assert_eq!(wf.pending_count(), 1);
    }

    #[test]
    fn test_reject_changes_status() {
        let mut wf = ClearanceWorkflow::new();
        wf.submit(make_req("r1", false));
        assert!(wf.reject("r1"));
        assert_eq!(
            wf.lookup("r1")
                .expect("rights test operation should succeed")
                .status,
            ClearanceStatus::Rejected
        );
    }

    #[test]
    fn test_approve_missing_returns_false() {
        let mut wf = ClearanceWorkflow::new();
        assert!(!wf.approve("ghost"));
    }

    #[test]
    fn test_reject_missing_returns_false() {
        let mut wf = ClearanceWorkflow::new();
        assert!(!wf.reject("ghost"));
    }

    #[test]
    fn test_escalate_changes_status() {
        let mut wf = ClearanceWorkflow::new();
        wf.submit(make_req("r1", false));
        assert!(wf.escalate("r1"));
        assert_eq!(
            wf.lookup("r1")
                .expect("rights test operation should succeed")
                .status,
            ClearanceStatus::Escalated
        );
    }

    #[test]
    fn test_urgent_pending_filter() {
        let mut wf = ClearanceWorkflow::new();
        wf.submit(make_req("r1", true));
        wf.submit(make_req("r2", false));
        wf.submit(make_req("r3", true));
        let urgent = wf.urgent_pending();
        assert_eq!(urgent.len(), 2);
    }

    #[test]
    fn test_lookup_existing() {
        let mut wf = ClearanceWorkflow::new();
        wf.submit(make_req("r1", false));
        assert!(wf.lookup("r1").is_some());
    }

    #[test]
    fn test_lookup_missing_returns_none() {
        let wf = ClearanceWorkflow::new();
        assert!(wf.lookup("ghost").is_none());
    }
}
