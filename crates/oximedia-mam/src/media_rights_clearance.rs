//! Media rights clearance workflow.
//!
//! Tracks the end-to-end clearance process for media rights: from initial
//! rights identification through territory and usage-type licensing, right
//! through to final clearance sign-off.
//!
//! # Key concepts
//!
//! * `RightsHolder` – a named entity that owns or administers rights.
//! * `UsageType` – how the asset will be used (broadcast, streaming, sync, etc.).
//! * `Territory` – geographic scope of the rights claim.
//! * `ClearanceStatus` – lifecycle state of a single clearance request.
//! * `ClearanceRequest` – a request to clear specific rights from a holder.
//! * `ClearanceRecord` – persisted record of a granted/denied clearance.
//! * `ClearanceWorkflow` – orchestrates multi-party clearances for an asset.
//! * `ClearanceRegistry` – stores all workflows indexed by asset id.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Territory
// ---------------------------------------------------------------------------

/// Geographic scope for a rights claim.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Territory {
    /// World-wide rights.
    Worldwide,
    /// ISO 3166-1 alpha-2 country code (e.g. `"US"`, `"GB"`).
    Country(String),
    /// Named regional grouping (e.g. `"EU"`, `"APAC"`, `"LATAM"`).
    Region(String),
    /// All territories except a specific exclusion list.
    WorldwideExcluding(Vec<String>),
}

impl Territory {
    /// Human-readable display string.
    #[must_use]
    pub fn display(&self) -> String {
        match self {
            Self::Worldwide => "Worldwide".to_string(),
            Self::Country(c) => c.clone(),
            Self::Region(r) => r.clone(),
            Self::WorldwideExcluding(excl) => format!("Worldwide (excl. {})", excl.join(", ")),
        }
    }

    /// Returns `true` if the territory includes the given ISO country code.
    #[must_use]
    pub fn includes_country(&self, country: &str) -> bool {
        match self {
            Self::Worldwide => true,
            Self::Country(c) => c.eq_ignore_ascii_case(country),
            Self::Region(_) => false,
            Self::WorldwideExcluding(excl) => !excl
                .iter()
                .any(|e| e.eq_ignore_ascii_case(country)),
        }
    }
}

// ---------------------------------------------------------------------------
// UsageType
// ---------------------------------------------------------------------------

/// The type of usage being cleared.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UsageType {
    /// Linear TV / broadcast transmission.
    Broadcast,
    /// Subscription or ad-supported streaming (SVOD / AVOD).
    Streaming,
    /// Synchronisation with visual media (ads, film, trailers).
    Synchronisation,
    /// Physical / digital distribution (DVD, download-to-own).
    Distribution,
    /// Educational or institutional use.
    Educational,
    /// Archival or research use only.
    Archival,
    /// Social media re-use.
    SocialMedia,
    /// Public performance (cinema, venue).
    PublicPerformance,
    /// Custom usage type.
    Custom(String),
}

impl UsageType {
    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::Broadcast => "Broadcast",
            Self::Streaming => "Streaming",
            Self::Synchronisation => "Synchronisation",
            Self::Distribution => "Distribution",
            Self::Educational => "Educational",
            Self::Archival => "Archival",
            Self::SocialMedia => "Social Media",
            Self::PublicPerformance => "Public Performance",
            Self::Custom(s) => s.as_str(),
        }
    }
}

// ---------------------------------------------------------------------------
// RightsHolder
// ---------------------------------------------------------------------------

/// A named entity that holds or administers rights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RightsHolder {
    /// Unique identifier.
    pub id: Uuid,
    /// Legal or display name.
    pub name: String,
    /// Contact email for clearance requests.
    pub contact_email: Option<String>,
    /// IPI (Interested Parties Information) code, if applicable.
    pub ipi_code: Option<String>,
    /// ISNI (International Standard Name Identifier), if applicable.
    pub isni: Option<String>,
    /// Free-form notes.
    pub notes: Option<String>,
    /// When this holder record was created.
    pub created_at: DateTime<Utc>,
}

impl RightsHolder {
    /// Create a new rights holder with only a name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            contact_email: None,
            ipi_code: None,
            isni: None,
            notes: None,
            created_at: Utc::now(),
        }
    }

    /// Builder: set the contact email.
    #[must_use]
    pub fn with_email(mut self, email: impl Into<String>) -> Self {
        self.contact_email = Some(email.into());
        self
    }

    /// Builder: set the IPI code.
    #[must_use]
    pub fn with_ipi(mut self, ipi: impl Into<String>) -> Self {
        self.ipi_code = Some(ipi.into());
        self
    }
}

// ---------------------------------------------------------------------------
// ClearanceStatus
// ---------------------------------------------------------------------------

/// Lifecycle state of a rights clearance request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClearanceStatus {
    /// Request has been drafted but not yet submitted.
    Draft,
    /// Request has been sent to the rights holder.
    Submitted,
    /// Rights holder has acknowledged the request.
    Acknowledged,
    /// Clearance has been granted.
    Cleared,
    /// Clearance has been denied outright.
    Denied,
    /// A counter-offer has been received (different terms / fee).
    CounterOffer,
    /// Clearance has been withdrawn after initially being granted.
    Withdrawn,
    /// Request has been cancelled.
    Cancelled,
}

impl ClearanceStatus {
    /// Returns `true` if the clearance is in a terminal state.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Cleared | Self::Denied | Self::Withdrawn | Self::Cancelled
        )
    }

    /// Returns `true` if the clearance can be used for publishing.
    #[must_use]
    pub fn is_cleared(&self) -> bool {
        matches!(self, Self::Cleared)
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::Draft => "Draft",
            Self::Submitted => "Submitted",
            Self::Acknowledged => "Acknowledged",
            Self::Cleared => "Cleared",
            Self::Denied => "Denied",
            Self::CounterOffer => "Counter Offer",
            Self::Withdrawn => "Withdrawn",
            Self::Cancelled => "Cancelled",
        }
    }
}

// ---------------------------------------------------------------------------
// ClearanceFee
// ---------------------------------------------------------------------------

/// A monetary fee associated with a clearance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClearanceFee {
    /// Amount in the smallest currency unit (e.g. cents).
    pub amount_minor: i64,
    /// ISO 4217 currency code (e.g. `"USD"`, `"EUR"`).
    pub currency: String,
    /// Whether the fee has been paid.
    pub paid: bool,
    /// When the fee was paid.
    pub paid_at: Option<DateTime<Utc>>,
}

impl ClearanceFee {
    /// Create a new unpaid fee.
    #[must_use]
    pub fn new(amount_minor: i64, currency: impl Into<String>) -> Self {
        Self {
            amount_minor,
            currency: currency.into(),
            paid: false,
            paid_at: None,
        }
    }

    /// Mark the fee as paid at the given timestamp.
    pub fn mark_paid(&mut self, at: DateTime<Utc>) {
        self.paid = true;
        self.paid_at = Some(at);
    }

    /// Formatted amount string (e.g. `"USD 10.50"`).
    #[must_use]
    pub fn display(&self) -> String {
        let dollars = self.amount_minor / 100;
        let cents = (self.amount_minor % 100).unsigned_abs();
        format!("{} {dollars}.{cents:02}", self.currency)
    }
}

// ---------------------------------------------------------------------------
// ClearanceRequest
// ---------------------------------------------------------------------------

/// A request to clear a specific set of rights from a holder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClearanceRequest {
    /// Unique request id.
    pub id: Uuid,
    /// Asset this request is for.
    pub asset_id: Uuid,
    /// Rights holder being contacted.
    pub holder_id: Uuid,
    /// Usage types being cleared.
    pub usage_types: Vec<UsageType>,
    /// Geographic territory for the clearance.
    pub territory: Territory,
    /// Start of the licensed period (inclusive).
    pub license_start: Option<DateTime<Utc>>,
    /// End of the licensed period (inclusive); `None` = perpetual.
    pub license_end: Option<DateTime<Utc>>,
    /// Current clearance status.
    pub status: ClearanceStatus,
    /// Fee agreed for this clearance, if any.
    pub fee: Option<ClearanceFee>,
    /// Free-form note attached to this request.
    pub notes: Option<String>,
    /// ISO 639-1 language code of the content being cleared (if applicable).
    pub language: Option<String>,
    /// When the request was created.
    pub created_at: DateTime<Utc>,
    /// When the request was last updated.
    pub updated_at: DateTime<Utc>,
    /// Status history log.
    pub history: Vec<ClearanceHistoryEntry>,
}

impl ClearanceRequest {
    /// Create a new draft clearance request.
    #[must_use]
    pub fn new(
        asset_id: Uuid,
        holder_id: Uuid,
        usage_types: Vec<UsageType>,
        territory: Territory,
    ) -> Self {
        let now = Utc::now();
        let id = Uuid::new_v4();
        Self {
            id,
            asset_id,
            holder_id,
            usage_types,
            territory,
            license_start: None,
            license_end: None,
            status: ClearanceStatus::Draft,
            fee: None,
            notes: None,
            language: None,
            created_at: now,
            updated_at: now,
            history: vec![ClearanceHistoryEntry {
                timestamp: now,
                from_status: None,
                to_status: ClearanceStatus::Draft,
                actor: None,
                note: Some("Request created".to_string()),
            }],
        }
    }

    /// Transition to a new status, recording the change in history.
    ///
    /// # Errors
    ///
    /// Returns `ClearanceError::InvalidTransition` if the transition is not
    /// allowed from the current state.
    pub fn transition(
        &mut self,
        new_status: ClearanceStatus,
        actor: Option<Uuid>,
        note: Option<String>,
    ) -> Result<(), ClearanceError> {
        if self.status.is_terminal() {
            return Err(ClearanceError::InvalidTransition {
                from: self.status.label().to_string(),
                to: new_status.label().to_string(),
            });
        }
        let old = self.status.clone();
        self.status = new_status.clone();
        self.updated_at = Utc::now();
        self.history.push(ClearanceHistoryEntry {
            timestamp: self.updated_at,
            from_status: Some(old),
            to_status: new_status,
            actor,
            note,
        });
        Ok(())
    }

    /// Returns `true` if this clearance covers the given country.
    #[must_use]
    pub fn covers_country(&self, country: &str) -> bool {
        self.territory.includes_country(country)
    }

    /// Returns `true` if this clearance is currently active (cleared and within
    /// the license window).
    #[must_use]
    pub fn is_active(&self, at: DateTime<Utc>) -> bool {
        if !self.status.is_cleared() {
            return false;
        }
        if let Some(start) = self.license_start {
            if at < start {
                return false;
            }
        }
        if let Some(end) = self.license_end {
            if at > end {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// ClearanceHistoryEntry
// ---------------------------------------------------------------------------

/// A single audit entry recording a status transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClearanceHistoryEntry {
    /// When the transition occurred.
    pub timestamp: DateTime<Utc>,
    /// Status before the transition, if any.
    pub from_status: Option<ClearanceStatus>,
    /// Status after the transition.
    pub to_status: ClearanceStatus,
    /// User who triggered the transition.
    pub actor: Option<Uuid>,
    /// Optional note explaining the transition.
    pub note: Option<String>,
}

// ---------------------------------------------------------------------------
// ClearanceWorkflow
// ---------------------------------------------------------------------------

/// Orchestrates multi-party clearances for a single asset.
///
/// An asset may require multiple clearances (e.g. master recording rights from
/// a label AND sync rights from a publisher).  The workflow is only considered
/// fully cleared when every required request reaches `Cleared` status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClearanceWorkflow {
    /// Unique workflow id.
    pub id: Uuid,
    /// The asset being cleared.
    pub asset_id: Uuid,
    /// Title / display name for this workflow.
    pub title: String,
    /// Individual clearance requests.
    pub requests: Vec<ClearanceRequest>,
    /// When the workflow was created.
    pub created_at: DateTime<Utc>,
    /// When the workflow was last updated.
    pub updated_at: DateTime<Utc>,
}

impl ClearanceWorkflow {
    /// Create a new empty workflow for an asset.
    #[must_use]
    pub fn new(asset_id: Uuid, title: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            asset_id,
            title: title.into(),
            requests: Vec::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Add a clearance request to this workflow.
    pub fn add_request(&mut self, request: ClearanceRequest) {
        self.updated_at = Utc::now();
        self.requests.push(request);
    }

    /// Returns `true` when every request is in the `Cleared` state.
    #[must_use]
    pub fn is_fully_cleared(&self) -> bool {
        !self.requests.is_empty()
            && self
                .requests
                .iter()
                .all(|r| r.status == ClearanceStatus::Cleared)
    }

    /// Returns requests that are still pending (not in a terminal state).
    #[must_use]
    pub fn pending_requests(&self) -> Vec<&ClearanceRequest> {
        self.requests
            .iter()
            .filter(|r| !r.status.is_terminal())
            .collect()
    }

    /// Returns requests in the `Denied` state.
    #[must_use]
    pub fn denied_requests(&self) -> Vec<&ClearanceRequest> {
        self.requests
            .iter()
            .filter(|r| r.status == ClearanceStatus::Denied)
            .collect()
    }

    /// Returns the total estimated cost of all fees across cleared requests.
    #[must_use]
    pub fn total_cleared_cost_minor(&self) -> i64 {
        self.requests
            .iter()
            .filter(|r| r.status.is_cleared())
            .filter_map(|r| r.fee.as_ref())
            .map(|f| f.amount_minor)
            .sum()
    }

    /// Returns the subset of requests that cover a specific country.
    #[must_use]
    pub fn requests_for_country(&self, country: &str) -> Vec<&ClearanceRequest> {
        self.requests
            .iter()
            .filter(|r| r.covers_country(country))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ClearanceError
// ---------------------------------------------------------------------------

/// Errors that can arise during rights clearance operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClearanceError {
    /// A status transition is not valid from the current state.
    InvalidTransition { from: String, to: String },
    /// A requested clearance request was not found.
    RequestNotFound(Uuid),
    /// A workflow was not found.
    WorkflowNotFound(Uuid),
    /// A rights holder was not found.
    HolderNotFound(Uuid),
    /// Business logic violation.
    Validation(String),
}

impl std::fmt::Display for ClearanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTransition { from, to } => {
                write!(f, "Invalid transition from '{from}' to '{to}'")
            }
            Self::RequestNotFound(id) => write!(f, "Clearance request not found: {id}"),
            Self::WorkflowNotFound(id) => write!(f, "Clearance workflow not found: {id}"),
            Self::HolderNotFound(id) => write!(f, "Rights holder not found: {id}"),
            Self::Validation(msg) => write!(f, "Validation error: {msg}"),
        }
    }
}

impl std::error::Error for ClearanceError {}

// ---------------------------------------------------------------------------
// ClearanceRegistry
// ---------------------------------------------------------------------------

/// In-memory store of clearance workflows and rights holders.
///
/// In production this would be backed by a persistent database; the registry
/// here is suitable for unit tests and lightweight usage without a DB.
#[derive(Debug, Default)]
pub struct ClearanceRegistry {
    /// Workflows indexed by their id.
    workflows: HashMap<Uuid, ClearanceWorkflow>,
    /// Rights holders indexed by their id.
    holders: HashMap<Uuid, RightsHolder>,
    /// Secondary index: asset_id → list of workflow ids.
    by_asset: HashMap<Uuid, Vec<Uuid>>,
}

impl ClearanceRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a rights holder.
    pub fn register_holder(&mut self, holder: RightsHolder) {
        self.holders.insert(holder.id, holder);
    }

    /// Look up a rights holder by id.
    ///
    /// # Errors
    ///
    /// Returns `ClearanceError::HolderNotFound` if the id is unknown.
    pub fn get_holder(&self, id: Uuid) -> Result<&RightsHolder, ClearanceError> {
        self.holders
            .get(&id)
            .ok_or(ClearanceError::HolderNotFound(id))
    }

    /// Add a workflow to the registry.
    pub fn add_workflow(&mut self, workflow: ClearanceWorkflow) {
        let asset_id = workflow.asset_id;
        let wf_id = workflow.id;
        self.workflows.insert(wf_id, workflow);
        self.by_asset.entry(asset_id).or_default().push(wf_id);
    }

    /// Get a workflow by id.
    ///
    /// # Errors
    ///
    /// Returns `ClearanceError::WorkflowNotFound` if not found.
    pub fn get_workflow(&self, id: Uuid) -> Result<&ClearanceWorkflow, ClearanceError> {
        self.workflows
            .get(&id)
            .ok_or(ClearanceError::WorkflowNotFound(id))
    }

    /// Get a mutable workflow by id.
    ///
    /// # Errors
    ///
    /// Returns `ClearanceError::WorkflowNotFound` if not found.
    pub fn get_workflow_mut(
        &mut self,
        id: Uuid,
    ) -> Result<&mut ClearanceWorkflow, ClearanceError> {
        self.workflows
            .get_mut(&id)
            .ok_or(ClearanceError::WorkflowNotFound(id))
    }

    /// Return all workflows for a specific asset.
    #[must_use]
    pub fn workflows_for_asset(&self, asset_id: Uuid) -> Vec<&ClearanceWorkflow> {
        self.by_asset
            .get(&asset_id)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.workflows.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Returns `true` if the asset has at least one fully-cleared workflow.
    #[must_use]
    pub fn asset_is_cleared(&self, asset_id: Uuid) -> bool {
        self.workflows_for_asset(asset_id)
            .iter()
            .any(|wf| wf.is_fully_cleared())
    }

    /// Count of registered rights holders.
    #[must_use]
    pub fn holder_count(&self) -> usize {
        self.holders.len()
    }

    /// Count of registered workflows.
    #[must_use]
    pub fn workflow_count(&self) -> usize {
        self.workflows.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_holder() -> RightsHolder {
        RightsHolder::new("Acme Music Publishing")
            .with_email("rights@acme.example")
            .with_ipi("00012345678")
    }

    fn make_request(asset_id: Uuid, holder_id: Uuid) -> ClearanceRequest {
        ClearanceRequest::new(
            asset_id,
            holder_id,
            vec![UsageType::Broadcast, UsageType::Streaming],
            Territory::Country("US".to_string()),
        )
    }

    #[test]
    fn test_territory_display_worldwide() {
        assert_eq!(Territory::Worldwide.display(), "Worldwide");
    }

    #[test]
    fn test_territory_includes_country_direct() {
        let t = Territory::Country("US".to_string());
        assert!(t.includes_country("US"));
        assert!(!t.includes_country("GB"));
    }

    #[test]
    fn test_territory_worldwide_includes_any_country() {
        assert!(Territory::Worldwide.includes_country("JP"));
        assert!(Territory::Worldwide.includes_country("DE"));
    }

    #[test]
    fn test_territory_worldwide_excluding() {
        let t = Territory::WorldwideExcluding(vec!["CN".to_string(), "RU".to_string()]);
        assert!(t.includes_country("US"));
        assert!(!t.includes_country("CN"));
        assert!(!t.includes_country("RU"));
    }

    #[test]
    fn test_clearance_status_is_terminal() {
        assert!(ClearanceStatus::Cleared.is_terminal());
        assert!(ClearanceStatus::Denied.is_terminal());
        assert!(!ClearanceStatus::Submitted.is_terminal());
        assert!(!ClearanceStatus::Draft.is_terminal());
    }

    #[test]
    fn test_clearance_fee_display() {
        let fee = ClearanceFee::new(1050, "USD");
        assert_eq!(fee.display(), "USD 10.50");
    }

    #[test]
    fn test_clearance_fee_mark_paid() {
        let mut fee = ClearanceFee::new(500, "EUR");
        assert!(!fee.paid);
        let now = Utc::now();
        fee.mark_paid(now);
        assert!(fee.paid);
        assert_eq!(fee.paid_at, Some(now));
    }

    #[test]
    fn test_clearance_request_transition_success() {
        let asset_id = Uuid::new_v4();
        let holder_id = Uuid::new_v4();
        let mut req = make_request(asset_id, holder_id);
        assert_eq!(req.status, ClearanceStatus::Draft);
        req.transition(ClearanceStatus::Submitted, None, None)
            .expect("transition should succeed");
        assert_eq!(req.status, ClearanceStatus::Submitted);
        assert_eq!(req.history.len(), 2);
    }

    #[test]
    fn test_clearance_request_transition_from_terminal_fails() {
        let asset_id = Uuid::new_v4();
        let holder_id = Uuid::new_v4();
        let mut req = make_request(asset_id, holder_id);
        req.status = ClearanceStatus::Cleared;
        let result = req.transition(ClearanceStatus::Denied, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_clearance_request_is_active() {
        let asset_id = Uuid::new_v4();
        let holder_id = Uuid::new_v4();
        let mut req = make_request(asset_id, holder_id);
        req.status = ClearanceStatus::Cleared;
        assert!(req.is_active(Utc::now()));
    }

    #[test]
    fn test_workflow_is_fully_cleared() {
        let asset_id = Uuid::new_v4();
        let holder_id = Uuid::new_v4();
        let mut wf = ClearanceWorkflow::new(asset_id, "Test Workflow");
        let mut req = make_request(asset_id, holder_id);
        req.status = ClearanceStatus::Cleared;
        wf.add_request(req);
        assert!(wf.is_fully_cleared());
    }

    #[test]
    fn test_workflow_not_cleared_with_pending_request() {
        let asset_id = Uuid::new_v4();
        let holder_id = Uuid::new_v4();
        let mut wf = ClearanceWorkflow::new(asset_id, "Test Workflow");
        let mut req1 = make_request(asset_id, holder_id);
        req1.status = ClearanceStatus::Cleared;
        let req2 = make_request(asset_id, holder_id);
        // req2 remains in Draft
        wf.add_request(req1);
        wf.add_request(req2);
        assert!(!wf.is_fully_cleared());
    }

    #[test]
    fn test_workflow_total_cleared_cost() {
        let asset_id = Uuid::new_v4();
        let holder_id = Uuid::new_v4();
        let mut wf = ClearanceWorkflow::new(asset_id, "Cost Test");
        let mut req = make_request(asset_id, holder_id);
        req.status = ClearanceStatus::Cleared;
        req.fee = Some(ClearanceFee::new(2000, "USD"));
        wf.add_request(req);
        assert_eq!(wf.total_cleared_cost_minor(), 2000);
    }

    #[test]
    fn test_registry_add_and_retrieve_workflow() {
        let asset_id = Uuid::new_v4();
        let mut registry = ClearanceRegistry::new();
        let wf = ClearanceWorkflow::new(asset_id, "Registry Test");
        let wf_id = wf.id;
        registry.add_workflow(wf);
        let retrieved = registry.get_workflow(wf_id).expect("should exist");
        assert_eq!(retrieved.asset_id, asset_id);
    }

    #[test]
    fn test_registry_workflows_for_asset() {
        let asset_id = Uuid::new_v4();
        let mut registry = ClearanceRegistry::new();
        registry.add_workflow(ClearanceWorkflow::new(asset_id, "WF1"));
        registry.add_workflow(ClearanceWorkflow::new(asset_id, "WF2"));
        let wfs = registry.workflows_for_asset(asset_id);
        assert_eq!(wfs.len(), 2);
    }

    #[test]
    fn test_registry_asset_is_cleared() {
        let asset_id = Uuid::new_v4();
        let holder_id = Uuid::new_v4();
        let mut registry = ClearanceRegistry::new();
        let mut wf = ClearanceWorkflow::new(asset_id, "Cleared WF");
        let mut req = make_request(asset_id, holder_id);
        req.status = ClearanceStatus::Cleared;
        wf.add_request(req);
        registry.add_workflow(wf);
        assert!(registry.asset_is_cleared(asset_id));
    }

    #[test]
    fn test_rights_holder_construction() {
        let holder = make_holder();
        assert_eq!(holder.name, "Acme Music Publishing");
        assert_eq!(
            holder.contact_email,
            Some("rights@acme.example".to_string())
        );
        assert_eq!(holder.ipi_code, Some("00012345678".to_string()));
    }

    #[test]
    fn test_usage_type_label() {
        assert_eq!(UsageType::Broadcast.label(), "Broadcast");
        assert_eq!(UsageType::Streaming.label(), "Streaming");
        assert_eq!(
            UsageType::Custom("Podcast".to_string()).label(),
            "Podcast"
        );
    }
}
