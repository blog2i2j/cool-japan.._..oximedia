//! Archive restore workflows: restore requests, SLA tracking, and
//! priority-based scheduling.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Priority level for a restore request.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum RestorePriority {
    /// Low – best effort, may take days
    Low = 0,
    /// Normal – deliver within defined SLA window
    Normal = 1,
    /// High – expedited restore
    High = 2,
    /// Critical – emergency, first in queue
    Critical = 3,
}

impl RestorePriority {
    /// Expected delivery window in seconds.
    #[must_use]
    pub fn sla_secs(&self) -> u64 {
        match self {
            Self::Low => 3 * 24 * 3600,
            Self::Normal => 12 * 3600,
            Self::High => 3 * 3600,
            Self::Critical => 3600,
        }
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Normal => "normal",
            Self::High => "high",
            Self::Critical => "critical",
        }
    }
}

/// Status of a restore request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RestoreStatus {
    /// Queued, not yet started
    Pending,
    /// Restore is in progress
    InProgress,
    /// Restore completed successfully
    Completed,
    /// Restore failed
    Failed,
    /// Cancelled by user
    Cancelled,
}

/// A single restore request.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RestoreRequest {
    /// Unique request identifier
    pub request_id: String,
    /// Asset being restored
    pub asset_id: String,
    /// Priority level
    pub priority: RestorePriority,
    /// Current status
    pub status: RestoreStatus,
    /// Unix timestamp when request was created
    pub created_at: u64,
    /// Requester name or system identifier
    pub requester: String,
    /// Size of the asset in bytes
    pub size_bytes: u64,
}

impl RestoreRequest {
    /// Create a new pending restore request.
    #[must_use]
    pub fn new(
        request_id: impl Into<String>,
        asset_id: impl Into<String>,
        priority: RestorePriority,
        requester: impl Into<String>,
        size_bytes: u64,
        created_at: u64,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            asset_id: asset_id.into(),
            priority,
            status: RestoreStatus::Pending,
            created_at,
            requester: requester.into(),
            size_bytes,
        }
    }

    /// SLA deadline (created_at + priority window).
    #[must_use]
    pub fn sla_deadline(&self) -> u64 {
        self.created_at + self.priority.sla_secs()
    }

    /// Whether the SLA is breached given current time.
    #[must_use]
    pub fn is_sla_breached(&self, now: u64) -> bool {
        self.status != RestoreStatus::Completed && now > self.sla_deadline()
    }
}

/// Entry wrapper for the priority queue.
#[derive(Debug, Eq, PartialEq)]
struct QueueEntry {
    priority: RestorePriority,
    created_at: u64,
    request_id: String,
}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher priority first; for equal priority, earlier created_at first
        self.priority
            .cmp(&other.priority)
            .then(Reverse(self.created_at).cmp(&Reverse(other.created_at)))
    }
}

/// Manages restore requests and SLA tracking.
#[derive(Debug, Default)]
pub struct RestoreWorkflowManager {
    requests: std::collections::HashMap<String, RestoreRequest>,
    queue: BinaryHeap<QueueEntry>,
}

impl RestoreWorkflowManager {
    /// Create a new manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Submit a restore request.
    pub fn submit(&mut self, req: RestoreRequest) {
        self.queue.push(QueueEntry {
            priority: req.priority,
            created_at: req.created_at,
            request_id: req.request_id.clone(),
        });
        self.requests.insert(req.request_id.clone(), req);
    }

    /// Pop the next highest-priority pending request.
    pub fn next_request(&mut self) -> Option<RestoreRequest> {
        while let Some(entry) = self.queue.pop() {
            if let Some(req) = self.requests.get(&entry.request_id) {
                if req.status == RestoreStatus::Pending {
                    return self.requests.get(&entry.request_id).cloned();
                }
            }
        }
        None
    }

    /// Update the status of a request.
    pub fn update_status(&mut self, request_id: &str, status: RestoreStatus) -> bool {
        if let Some(req) = self.requests.get_mut(request_id) {
            req.status = status;
            true
        } else {
            false
        }
    }

    /// Return all requests breaching their SLA at a given time.
    #[must_use]
    pub fn sla_breaches(&self, now: u64) -> Vec<&RestoreRequest> {
        self.requests
            .values()
            .filter(|r| r.is_sla_breached(now))
            .collect()
    }

    /// Total number of tracked requests.
    #[must_use]
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    /// Whether the manager has no requests.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// Count requests by status.
    #[must_use]
    pub fn count_by_status(&self, status: RestoreStatus) -> usize {
        self.requests
            .values()
            .filter(|r| r.status == status)
            .count()
    }
}

/// SLA report summarising compliance across all requests.
#[derive(Debug, Clone)]
pub struct SlaReport {
    /// Total requests analysed
    pub total: usize,
    /// Number of SLA breaches
    pub breaches: usize,
    /// Compliance rate 0.0–1.0
    pub compliance_rate: f64,
}

impl SlaReport {
    /// Build a report from a manager at a given timestamp.
    #[must_use]
    pub fn build(manager: &RestoreWorkflowManager, now: u64) -> Self {
        let total = manager.len();
        let breaches = manager.sla_breaches(now).len();
        let compliance_rate = if total == 0 {
            1.0
        } else {
            (total - breaches) as f64 / total as f64
        };
        Self {
            total,
            breaches,
            compliance_rate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_req(id: &str, priority: RestorePriority, created_at: u64) -> RestoreRequest {
        RestoreRequest::new(
            id,
            "asset-001",
            priority,
            "test-user",
            1_000_000,
            created_at,
        )
    }

    #[test]
    fn test_restore_priority_ordering() {
        assert!(RestorePriority::Critical > RestorePriority::High);
        assert!(RestorePriority::High > RestorePriority::Normal);
        assert!(RestorePriority::Normal > RestorePriority::Low);
    }

    #[test]
    fn test_restore_priority_labels() {
        assert_eq!(RestorePriority::Low.label(), "low");
        assert_eq!(RestorePriority::Normal.label(), "normal");
        assert_eq!(RestorePriority::High.label(), "high");
        assert_eq!(RestorePriority::Critical.label(), "critical");
    }

    #[test]
    fn test_sla_secs_ordering() {
        assert!(RestorePriority::Low.sla_secs() > RestorePriority::High.sla_secs());
        assert!(RestorePriority::Critical.sla_secs() < RestorePriority::Normal.sla_secs());
    }

    #[test]
    fn test_restore_request_sla_deadline() {
        let req = make_req("r1", RestorePriority::Critical, 1_000_000);
        assert_eq!(
            req.sla_deadline(),
            1_000_000 + RestorePriority::Critical.sla_secs()
        );
    }

    #[test]
    fn test_restore_request_sla_not_breached() {
        let req = make_req("r2", RestorePriority::Critical, 1_000_000);
        // now = created + 100, well within 1h SLA
        assert!(!req.is_sla_breached(1_000_100));
    }

    #[test]
    fn test_restore_request_sla_breached() {
        let req = make_req("r3", RestorePriority::Critical, 0);
        // now = far in the future
        assert!(req.is_sla_breached(1_000_000_000));
    }

    #[test]
    fn test_manager_submit_and_len() {
        let mut mgr = RestoreWorkflowManager::new();
        mgr.submit(make_req("r4", RestorePriority::Normal, 1000));
        assert_eq!(mgr.len(), 1);
    }

    #[test]
    fn test_manager_next_request_priority_order() {
        let mut mgr = RestoreWorkflowManager::new();
        mgr.submit(make_req("low", RestorePriority::Low, 1000));
        mgr.submit(make_req("crit", RestorePriority::Critical, 1001));
        mgr.submit(make_req("norm", RestorePriority::Normal, 1002));
        let first = mgr.next_request().expect("operation should succeed");
        assert_eq!(first.priority, RestorePriority::Critical);
    }

    #[test]
    fn test_manager_update_status() {
        let mut mgr = RestoreWorkflowManager::new();
        mgr.submit(make_req("r5", RestorePriority::High, 0));
        let updated = mgr.update_status("r5", RestoreStatus::Completed);
        assert!(updated);
        assert_eq!(mgr.count_by_status(RestoreStatus::Completed), 1);
    }

    #[test]
    fn test_manager_update_status_missing() {
        let mut mgr = RestoreWorkflowManager::new();
        assert!(!mgr.update_status("nonexistent", RestoreStatus::Completed));
    }

    #[test]
    fn test_sla_breaches_detection() {
        let mut mgr = RestoreWorkflowManager::new();
        mgr.submit(make_req("old", RestorePriority::Critical, 0));
        let breaches = mgr.sla_breaches(999_999_999);
        assert_eq!(breaches.len(), 1);
    }

    #[test]
    fn test_sla_report_compliance() {
        let mut mgr = RestoreWorkflowManager::new();
        mgr.submit(make_req("ok", RestorePriority::Normal, 1_000_000));
        mgr.update_status("ok", RestoreStatus::Completed);
        let report = SlaReport::build(&mgr, 2_000_000);
        // completed request is not breached
        assert_eq!(report.breaches, 0);
        assert!((report.compliance_rate - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_sla_report_empty_manager() {
        let mgr = RestoreWorkflowManager::new();
        let report = SlaReport::build(&mgr, 0);
        assert_eq!(report.total, 0);
        assert!((report.compliance_rate - 1.0).abs() < 1e-9);
    }
}
