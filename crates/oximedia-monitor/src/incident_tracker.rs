#![allow(dead_code)]
//! Incident tracking and lifecycle management.
//!
//! Provides structures for creating, updating, resolving, and querying
//! incidents that arise from monitoring alerts or manual reports.

use std::collections::HashMap;

/// Severity level of an incident.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IncidentSeverity {
    /// Informational — no action required.
    Info,
    /// Warning — may require attention.
    Warning,
    /// Error — service degraded.
    Error,
    /// Critical — service outage.
    Critical,
}

impl std::fmt::Display for IncidentSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "info"),
            Self::Warning => write!(f, "warning"),
            Self::Error => write!(f, "error"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

/// Current state of an incident in its lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IncidentState {
    /// Newly created, not yet acknowledged.
    Open,
    /// Acknowledged by an operator.
    Acknowledged,
    /// Actively being investigated.
    Investigating,
    /// A fix or mitigation has been applied.
    Mitigated,
    /// Fully resolved.
    Resolved,
    /// Closed without resolution (e.g., false alarm).
    Closed,
}

impl std::fmt::Display for IncidentState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Open => write!(f, "open"),
            Self::Acknowledged => write!(f, "acknowledged"),
            Self::Investigating => write!(f, "investigating"),
            Self::Mitigated => write!(f, "mitigated"),
            Self::Resolved => write!(f, "resolved"),
            Self::Closed => write!(f, "closed"),
        }
    }
}

/// A timestamped note on an incident.
#[derive(Debug, Clone)]
pub struct IncidentNote {
    /// Unix timestamp when the note was added.
    pub timestamp: f64,
    /// Author of the note.
    pub author: String,
    /// Note text.
    pub message: String,
}

impl IncidentNote {
    /// Create a new note.
    pub fn new(timestamp: f64, author: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            timestamp,
            author: author.into(),
            message: message.into(),
        }
    }
}

/// A single incident record.
#[derive(Debug, Clone)]
pub struct Incident {
    /// Unique identifier.
    pub id: u64,
    /// Human-readable title.
    pub title: String,
    /// Detailed description.
    pub description: String,
    /// Severity.
    pub severity: IncidentSeverity,
    /// Current state.
    pub state: IncidentState,
    /// Unix timestamp when the incident was created.
    pub created_at: f64,
    /// Unix timestamp of last state change.
    pub updated_at: f64,
    /// Unix timestamp when resolved/closed, if applicable.
    pub resolved_at: Option<f64>,
    /// Assignee name, if any.
    pub assignee: Option<String>,
    /// Tags for categorisation.
    pub tags: Vec<String>,
    /// Timeline of notes.
    pub notes: Vec<IncidentNote>,
}

impl Incident {
    /// Create a new incident.
    pub fn new(
        id: u64,
        title: impl Into<String>,
        description: impl Into<String>,
        severity: IncidentSeverity,
        created_at: f64,
    ) -> Self {
        Self {
            id,
            title: title.into(),
            description: description.into(),
            severity,
            state: IncidentState::Open,
            created_at,
            updated_at: created_at,
            resolved_at: None,
            assignee: None,
            tags: Vec::new(),
            notes: Vec::new(),
        }
    }

    /// Transition the incident to a new state.
    pub fn transition(&mut self, new_state: IncidentState, timestamp: f64) {
        self.state = new_state;
        self.updated_at = timestamp;
        if matches!(new_state, IncidentState::Resolved | IncidentState::Closed) {
            self.resolved_at = Some(timestamp);
        }
    }

    /// Assign the incident to an operator.
    pub fn assign(&mut self, assignee: impl Into<String>, timestamp: f64) {
        self.assignee = Some(assignee.into());
        self.updated_at = timestamp;
    }

    /// Add a tag.
    pub fn add_tag(&mut self, tag: impl Into<String>) {
        self.tags.push(tag.into());
    }

    /// Add a note.
    pub fn add_note(&mut self, note: IncidentNote) {
        self.updated_at = note.timestamp;
        self.notes.push(note);
    }

    /// Check if the incident is still active (not resolved or closed).
    #[must_use]
    pub fn is_active(&self) -> bool {
        !matches!(self.state, IncidentState::Resolved | IncidentState::Closed)
    }

    /// Compute the duration from creation to resolution (or to `now`).
    #[must_use]
    pub fn duration(&self, now: f64) -> f64 {
        let end = self.resolved_at.unwrap_or(now);
        (end - self.created_at).max(0.0)
    }
}

/// Tracker that manages a collection of incidents.
#[derive(Debug)]
pub struct IncidentTracker {
    /// All incidents indexed by ID.
    incidents: HashMap<u64, Incident>,
    /// Auto-incrementing ID counter.
    next_id: u64,
}

impl IncidentTracker {
    /// Create a new empty tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            incidents: HashMap::new(),
            next_id: 1,
        }
    }

    /// Create and register a new incident, returning its ID.
    pub fn create(
        &mut self,
        title: impl Into<String>,
        description: impl Into<String>,
        severity: IncidentSeverity,
        created_at: f64,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let incident = Incident::new(id, title, description, severity, created_at);
        self.incidents.insert(id, incident);
        id
    }

    /// Get a reference to an incident by ID.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&Incident> {
        self.incidents.get(&id)
    }

    /// Get a mutable reference to an incident by ID.
    pub fn get_mut(&mut self, id: u64) -> Option<&mut Incident> {
        self.incidents.get_mut(&id)
    }

    /// Transition an incident's state.
    pub fn transition(&mut self, id: u64, state: IncidentState, timestamp: f64) -> bool {
        if let Some(inc) = self.incidents.get_mut(&id) {
            inc.transition(state, timestamp);
            true
        } else {
            false
        }
    }

    /// List all active incidents.
    #[must_use]
    pub fn active_incidents(&self) -> Vec<&Incident> {
        self.incidents.values().filter(|i| i.is_active()).collect()
    }

    /// List incidents filtered by severity.
    #[must_use]
    pub fn by_severity(&self, severity: IncidentSeverity) -> Vec<&Incident> {
        self.incidents
            .values()
            .filter(|i| i.severity == severity)
            .collect()
    }

    /// List incidents filtered by state.
    #[must_use]
    pub fn by_state(&self, state: IncidentState) -> Vec<&Incident> {
        self.incidents
            .values()
            .filter(|i| i.state == state)
            .collect()
    }

    /// Get total incident count.
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.incidents.len()
    }

    /// Get active incident count.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.incidents.values().filter(|i| i.is_active()).count()
    }

    /// Compute mean time to resolve for all resolved incidents.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn mean_time_to_resolve(&self) -> Option<f64> {
        let resolved: Vec<&Incident> = self
            .incidents
            .values()
            .filter(|i| i.resolved_at.is_some())
            .collect();
        if resolved.is_empty() {
            return None;
        }
        let total: f64 = resolved
            .iter()
            .map(|i| i.resolved_at.unwrap_or(i.created_at) - i.created_at)
            .sum();
        Some(total / resolved.len() as f64)
    }

    /// Remove resolved/closed incidents older than `cutoff` timestamp.
    pub fn purge_before(&mut self, cutoff: f64) -> usize {
        let before = self.incidents.len();
        self.incidents
            .retain(|_, i| i.is_active() || i.resolved_at.map_or(true, |t| t >= cutoff));
        before - self.incidents.len()
    }
}

impl Default for IncidentTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_severity_ordering() {
        assert!(IncidentSeverity::Info < IncidentSeverity::Warning);
        assert!(IncidentSeverity::Warning < IncidentSeverity::Error);
        assert!(IncidentSeverity::Error < IncidentSeverity::Critical);
    }

    #[test]
    fn test_severity_display() {
        assert_eq!(IncidentSeverity::Critical.to_string(), "critical");
        assert_eq!(IncidentSeverity::Info.to_string(), "info");
    }

    #[test]
    fn test_state_display() {
        assert_eq!(IncidentState::Open.to_string(), "open");
        assert_eq!(IncidentState::Resolved.to_string(), "resolved");
    }

    #[test]
    fn test_incident_creation() {
        let inc = Incident::new(1, "Test", "Description", IncidentSeverity::Warning, 1000.0);
        assert_eq!(inc.id, 1);
        assert_eq!(inc.state, IncidentState::Open);
        assert!(inc.is_active());
        assert!(inc.resolved_at.is_none());
    }

    #[test]
    fn test_incident_transition() {
        let mut inc = Incident::new(1, "Test", "Desc", IncidentSeverity::Error, 1000.0);
        inc.transition(IncidentState::Acknowledged, 1010.0);
        assert_eq!(inc.state, IncidentState::Acknowledged);
        assert!(inc.is_active());

        inc.transition(IncidentState::Resolved, 1050.0);
        assert_eq!(inc.state, IncidentState::Resolved);
        assert!(!inc.is_active());
        assert!(
            (inc.resolved_at.expect("resolved_at should be valid") - 1050.0).abs() < f64::EPSILON
        );
    }

    #[test]
    fn test_incident_assign() {
        let mut inc = Incident::new(1, "Test", "Desc", IncidentSeverity::Info, 1000.0);
        inc.assign("alice", 1005.0);
        assert_eq!(inc.assignee.as_deref(), Some("alice"));
    }

    #[test]
    fn test_incident_notes() {
        let mut inc = Incident::new(1, "Test", "Desc", IncidentSeverity::Info, 1000.0);
        inc.add_note(IncidentNote::new(1001.0, "bob", "Looking into it"));
        assert_eq!(inc.notes.len(), 1);
        assert_eq!(inc.notes[0].author, "bob");
    }

    #[test]
    fn test_incident_duration() {
        let mut inc = Incident::new(1, "Test", "Desc", IncidentSeverity::Error, 1000.0);
        assert!((inc.duration(1500.0) - 500.0).abs() < f64::EPSILON);
        inc.transition(IncidentState::Resolved, 1200.0);
        assert!((inc.duration(9999.0) - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tracker_create() {
        let mut tracker = IncidentTracker::new();
        let id = tracker.create("Outage", "DB down", IncidentSeverity::Critical, 1000.0);
        assert_eq!(id, 1);
        assert_eq!(tracker.total_count(), 1);
    }

    #[test]
    fn test_tracker_active_incidents() {
        let mut tracker = IncidentTracker::new();
        let id1 = tracker.create("A", "a", IncidentSeverity::Warning, 1000.0);
        let _id2 = tracker.create("B", "b", IncidentSeverity::Error, 1001.0);
        tracker.transition(id1, IncidentState::Resolved, 1050.0);

        assert_eq!(tracker.active_count(), 1);
        assert_eq!(tracker.active_incidents().len(), 1);
    }

    #[test]
    fn test_tracker_by_severity() {
        let mut tracker = IncidentTracker::new();
        tracker.create("A", "a", IncidentSeverity::Warning, 1000.0);
        tracker.create("B", "b", IncidentSeverity::Critical, 1001.0);
        tracker.create("C", "c", IncidentSeverity::Warning, 1002.0);

        assert_eq!(tracker.by_severity(IncidentSeverity::Warning).len(), 2);
        assert_eq!(tracker.by_severity(IncidentSeverity::Critical).len(), 1);
    }

    #[test]
    fn test_tracker_by_state() {
        let mut tracker = IncidentTracker::new();
        let id1 = tracker.create("A", "a", IncidentSeverity::Error, 1000.0);
        tracker.create("B", "b", IncidentSeverity::Error, 1001.0);
        tracker.transition(id1, IncidentState::Acknowledged, 1010.0);

        assert_eq!(tracker.by_state(IncidentState::Open).len(), 1);
        assert_eq!(tracker.by_state(IncidentState::Acknowledged).len(), 1);
    }

    #[test]
    fn test_mean_time_to_resolve() {
        let mut tracker = IncidentTracker::new();
        let id1 = tracker.create("A", "a", IncidentSeverity::Error, 1000.0);
        let id2 = tracker.create("B", "b", IncidentSeverity::Error, 1000.0);
        tracker.transition(id1, IncidentState::Resolved, 1100.0); // 100s
        tracker.transition(id2, IncidentState::Resolved, 1200.0); // 200s

        let mttr = tracker
            .mean_time_to_resolve()
            .expect("mean_time_to_resolve should succeed");
        assert!((mttr - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mean_time_to_resolve_none() {
        let tracker = IncidentTracker::new();
        assert!(tracker.mean_time_to_resolve().is_none());
    }

    #[test]
    fn test_purge_before() {
        let mut tracker = IncidentTracker::new();
        let id1 = tracker.create("old", "a", IncidentSeverity::Info, 100.0);
        let id2 = tracker.create("new", "b", IncidentSeverity::Info, 200.0);
        tracker.create("active", "c", IncidentSeverity::Error, 300.0);

        tracker.transition(id1, IncidentState::Resolved, 150.0);
        tracker.transition(id2, IncidentState::Resolved, 250.0);

        let purged = tracker.purge_before(200.0);
        assert_eq!(purged, 1); // only id1 resolved before 200
        assert_eq!(tracker.total_count(), 2);
    }

    #[test]
    fn test_incident_tags() {
        let mut inc = Incident::new(1, "Test", "Desc", IncidentSeverity::Info, 1000.0);
        inc.add_tag("database");
        inc.add_tag("production");
        assert_eq!(inc.tags.len(), 2);
        assert_eq!(inc.tags[0], "database");
    }
}
