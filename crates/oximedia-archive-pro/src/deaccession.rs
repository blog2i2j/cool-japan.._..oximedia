#![allow(dead_code)]
//! Deaccessioning management for archived digital assets.
//!
//! This module handles the controlled, policy-driven removal of items from the
//! archive. Deaccessioning requires approval workflows, audit trails, and
//! compliance checks before any asset can be permanently removed.

use std::collections::HashMap;
use std::time::SystemTime;

/// Reason for deaccessioning an archived asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeaccessionReason {
    /// Duplicate of another preserved asset.
    Duplicate,
    /// Retention period has expired.
    RetentionExpired,
    /// Rights or license have expired.
    RightsExpired,
    /// Asset is damaged beyond repair.
    IrreparableDamage,
    /// Superseded by a newer version.
    Superseded,
    /// Legal or regulatory requirement.
    LegalRequirement,
    /// Donor or depositor requested removal.
    DepositorRequest,
    /// Outside the scope of the collection.
    OutOfScope,
}

impl DeaccessionReason {
    /// Returns a human-readable label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Duplicate => "Duplicate asset",
            Self::RetentionExpired => "Retention period expired",
            Self::RightsExpired => "Rights/license expired",
            Self::IrreparableDamage => "Irreparable damage",
            Self::Superseded => "Superseded by newer version",
            Self::LegalRequirement => "Legal requirement",
            Self::DepositorRequest => "Depositor request",
            Self::OutOfScope => "Out of collection scope",
        }
    }

    /// Returns `true` if this reason requires mandatory legal review.
    #[must_use]
    pub const fn requires_legal_review(&self) -> bool {
        matches!(
            self,
            Self::LegalRequirement | Self::RightsExpired | Self::DepositorRequest
        )
    }
}

/// Current status of a deaccession request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeaccessionStatus {
    /// Request has been submitted.
    Pending,
    /// Under review by approvers.
    UnderReview,
    /// Approved and awaiting execution.
    Approved,
    /// Rejected by an approver.
    Rejected,
    /// Execution in progress (asset being removed).
    Executing,
    /// Completed successfully.
    Completed,
    /// Cancelled by the requester.
    Cancelled,
}

impl DeaccessionStatus {
    /// Returns `true` if this is a terminal state.
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        matches!(self, Self::Rejected | Self::Completed | Self::Cancelled)
    }

    /// Returns `true` if the request is still active.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        !self.is_terminal()
    }
}

/// A single approval decision.
#[derive(Debug, Clone)]
pub struct ApprovalDecision {
    /// The approver's name or identifier.
    pub approver: String,
    /// Whether the request was approved.
    pub approved: bool,
    /// Comments from the approver.
    pub comments: String,
    /// When the decision was made.
    pub decided_at: SystemTime,
}

/// A deaccession request for one or more assets.
#[derive(Debug, Clone)]
pub struct DeaccessionRequest {
    /// Unique identifier for the request.
    pub request_id: u64,
    /// Asset identifiers to deaccession.
    pub asset_ids: Vec<String>,
    /// Reason for deaccessioning.
    pub reason: DeaccessionReason,
    /// Detailed justification provided by the requester.
    pub justification: String,
    /// Who submitted the request.
    pub requester: String,
    /// When the request was created.
    pub created_at: SystemTime,
    /// Current status.
    pub status: DeaccessionStatus,
    /// Approval decisions collected.
    pub approvals: Vec<ApprovalDecision>,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
    /// Number of approvals required.
    pub required_approvals: u32,
}

impl DeaccessionRequest {
    /// Creates a new deaccession request.
    #[must_use]
    pub fn new(
        request_id: u64,
        asset_ids: Vec<String>,
        reason: DeaccessionReason,
        requester: &str,
        justification: &str,
    ) -> Self {
        Self {
            request_id,
            asset_ids,
            reason,
            justification: justification.to_string(),
            requester: requester.to_string(),
            created_at: SystemTime::now(),
            status: DeaccessionStatus::Pending,
            approvals: Vec::new(),
            metadata: HashMap::new(),
            required_approvals: 2,
        }
    }

    /// Sets the required number of approvals.
    #[must_use]
    pub const fn with_required_approvals(mut self, count: u32) -> Self {
        self.required_approvals = count;
        self
    }

    /// Records an approval decision.
    pub fn add_approval(&mut self, approver: &str, approved: bool, comments: &str) {
        self.approvals.push(ApprovalDecision {
            approver: approver.to_string(),
            approved,
            comments: comments.to_string(),
            decided_at: SystemTime::now(),
        });
        self.update_status();
    }

    /// Updates the status based on current approvals.
    fn update_status(&mut self) {
        if self.status.is_terminal() {
            return;
        }

        // Any rejection fails the request
        if self.approvals.iter().any(|a| !a.approved) {
            self.status = DeaccessionStatus::Rejected;
            return;
        }

        let approval_count = self.approvals.iter().filter(|a| a.approved).count();
        #[allow(clippy::cast_possible_truncation)]
        let count = approval_count as u32;
        if count >= self.required_approvals {
            self.status = DeaccessionStatus::Approved;
        } else {
            self.status = DeaccessionStatus::UnderReview;
        }
    }

    /// Returns the number of positive approvals received.
    #[must_use]
    pub fn approval_count(&self) -> usize {
        self.approvals.iter().filter(|a| a.approved).count()
    }

    /// Returns `true` if the request has enough approvals.
    #[must_use]
    pub fn is_fully_approved(&self) -> bool {
        self.status == DeaccessionStatus::Approved
    }

    /// Returns `true` if legal review is required based on the reason.
    #[must_use]
    pub fn needs_legal_review(&self) -> bool {
        self.reason.requires_legal_review()
    }

    /// Marks the request as executing.
    pub fn begin_execution(&mut self) -> bool {
        if self.status == DeaccessionStatus::Approved {
            self.status = DeaccessionStatus::Executing;
            true
        } else {
            false
        }
    }

    /// Marks the request as completed.
    pub fn complete(&mut self) -> bool {
        if self.status == DeaccessionStatus::Executing {
            self.status = DeaccessionStatus::Completed;
            true
        } else {
            false
        }
    }

    /// Cancels the request if it is not yet terminal.
    pub fn cancel(&mut self) -> bool {
        if self.status.is_active() {
            self.status = DeaccessionStatus::Cancelled;
            true
        } else {
            false
        }
    }
}

/// Registry that tracks all deaccession requests.
#[derive(Debug, Default)]
pub struct DeaccessionRegistry {
    /// All tracked requests.
    requests: Vec<DeaccessionRequest>,
    /// Next request ID.
    next_id: u64,
}

impl DeaccessionRegistry {
    /// Creates a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            requests: Vec::new(),
            next_id: 1,
        }
    }

    /// Submits a new deaccession request and returns its ID.
    pub fn submit(
        &mut self,
        asset_ids: Vec<String>,
        reason: DeaccessionReason,
        requester: &str,
        justification: &str,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.requests.push(DeaccessionRequest::new(
            id,
            asset_ids,
            reason,
            requester,
            justification,
        ));
        id
    }

    /// Looks up a request by ID.
    #[must_use]
    pub fn get(&self, request_id: u64) -> Option<&DeaccessionRequest> {
        self.requests.iter().find(|r| r.request_id == request_id)
    }

    /// Looks up a mutable request by ID.
    pub fn get_mut(&mut self, request_id: u64) -> Option<&mut DeaccessionRequest> {
        self.requests
            .iter_mut()
            .find(|r| r.request_id == request_id)
    }

    /// Returns all requests with a given status.
    #[must_use]
    pub fn by_status(&self, status: DeaccessionStatus) -> Vec<&DeaccessionRequest> {
        self.requests
            .iter()
            .filter(|r| r.status == status)
            .collect()
    }

    /// Returns the total number of requests.
    #[must_use]
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    /// Returns `true` if no requests exist.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reason_label() {
        assert_eq!(DeaccessionReason::Duplicate.label(), "Duplicate asset");
        assert_eq!(
            DeaccessionReason::Superseded.label(),
            "Superseded by newer version"
        );
    }

    #[test]
    fn test_reason_requires_legal() {
        assert!(DeaccessionReason::LegalRequirement.requires_legal_review());
        assert!(DeaccessionReason::RightsExpired.requires_legal_review());
        assert!(!DeaccessionReason::Duplicate.requires_legal_review());
    }

    #[test]
    fn test_status_terminal() {
        assert!(DeaccessionStatus::Completed.is_terminal());
        assert!(DeaccessionStatus::Rejected.is_terminal());
        assert!(DeaccessionStatus::Cancelled.is_terminal());
        assert!(!DeaccessionStatus::Pending.is_terminal());
        assert!(!DeaccessionStatus::Approved.is_terminal());
    }

    #[test]
    fn test_status_active() {
        assert!(DeaccessionStatus::Pending.is_active());
        assert!(DeaccessionStatus::UnderReview.is_active());
        assert!(!DeaccessionStatus::Completed.is_active());
    }

    #[test]
    fn test_new_request() {
        let req = DeaccessionRequest::new(
            1,
            vec!["asset-1".to_string()],
            DeaccessionReason::Duplicate,
            "admin",
            "Exact duplicate of asset-2",
        );
        assert_eq!(req.status, DeaccessionStatus::Pending);
        assert_eq!(req.asset_ids.len(), 1);
    }

    #[test]
    fn test_approval_workflow() {
        let mut req = DeaccessionRequest::new(
            1,
            vec!["a1".to_string()],
            DeaccessionReason::RetentionExpired,
            "user",
            "Retention expired",
        )
        .with_required_approvals(2);

        req.add_approval("approver1", true, "Looks good");
        assert_eq!(req.status, DeaccessionStatus::UnderReview);
        assert_eq!(req.approval_count(), 1);

        req.add_approval("approver2", true, "Approved");
        assert_eq!(req.status, DeaccessionStatus::Approved);
        assert!(req.is_fully_approved());
    }

    #[test]
    fn test_rejection() {
        let mut req = DeaccessionRequest::new(
            2,
            vec!["a2".to_string()],
            DeaccessionReason::OutOfScope,
            "user",
            "Out of scope",
        );
        req.add_approval("approver1", false, "Still relevant");
        assert_eq!(req.status, DeaccessionStatus::Rejected);
    }

    #[test]
    fn test_execution_lifecycle() {
        let mut req = DeaccessionRequest::new(
            3,
            vec!["a3".to_string()],
            DeaccessionReason::Duplicate,
            "user",
            "Dup",
        )
        .with_required_approvals(1);

        assert!(!req.begin_execution()); // not approved yet
        req.add_approval("approver1", true, "OK");
        assert!(req.begin_execution());
        assert_eq!(req.status, DeaccessionStatus::Executing);
        assert!(req.complete());
        assert_eq!(req.status, DeaccessionStatus::Completed);
    }

    #[test]
    fn test_cancel() {
        let mut req = DeaccessionRequest::new(
            4,
            vec!["a4".to_string()],
            DeaccessionReason::Superseded,
            "user",
            "Newer version available",
        );
        assert!(req.cancel());
        assert_eq!(req.status, DeaccessionStatus::Cancelled);
        assert!(!req.cancel()); // already terminal
    }

    #[test]
    fn test_registry_submit_and_lookup() {
        let mut registry = DeaccessionRegistry::new();
        let id = registry.submit(
            vec!["asset-x".to_string()],
            DeaccessionReason::Duplicate,
            "admin",
            "Duplicate found",
        );
        assert_eq!(id, 1);
        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());
        let req = registry.get(id).expect("operation should succeed");
        assert_eq!(req.requester, "admin");
    }

    #[test]
    fn test_registry_by_status() {
        let mut registry = DeaccessionRegistry::new();
        registry.submit(
            vec!["a1".to_string()],
            DeaccessionReason::Duplicate,
            "u1",
            "Dup",
        );
        registry.submit(
            vec!["a2".to_string()],
            DeaccessionReason::OutOfScope,
            "u2",
            "Scope",
        );
        let pending = registry.by_status(DeaccessionStatus::Pending);
        assert_eq!(pending.len(), 2);
        let approved = registry.by_status(DeaccessionStatus::Approved);
        assert_eq!(approved.len(), 0);
    }

    #[test]
    fn test_needs_legal_review() {
        let req = DeaccessionRequest::new(
            5,
            vec!["a5".to_string()],
            DeaccessionReason::LegalRequirement,
            "user",
            "Court order",
        );
        assert!(req.needs_legal_review());

        let req2 = DeaccessionRequest::new(
            6,
            vec!["a6".to_string()],
            DeaccessionReason::Duplicate,
            "user",
            "Dup",
        );
        assert!(!req2.needs_legal_review());
    }
}
