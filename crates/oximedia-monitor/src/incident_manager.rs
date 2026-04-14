//! Full incident lifecycle management with MTTD/MTTR analytics, post-mortem
//! workflows, SLI impact scoring, and on-call escalation chains.
//!
//! This module extends the basic [`crate::incident_tracker`] with a production-
//! grade incident management system that captures:
//!
//! - **Richer lifecycle states**: Detected → Triaged → Acknowledged → Investigating →
//!   Mitigated → Resolved → PostMortem → Closed.
//! - **MTTD** (Mean Time to Detect): from event start to first detection timestamp.
//! - **MTTR** (Mean Time to Recover): from detection to full resolution.
//! - **SLI impact** score: estimated user-visible impact (0–100) fed by the
//!   operator when opening or updating an incident.
//! - **Escalation chain**: ordered list of on-call contacts that are paged in
//!   sequence when an incident exceeds an escalation timeout.
//! - **Post-mortem**: a structured document attached to resolved incidents
//!   capturing root cause, action items, and contributing factors.

#![allow(dead_code)]

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Lifecycle State
// ---------------------------------------------------------------------------

/// Full lifecycle state of a managed incident.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum IncidentLifecycle {
    /// Alert fired; incident record created but not yet triaged.
    Detected,
    /// Initial triage complete; severity and scope assessed.
    Triaged,
    /// An on-call engineer has acknowledged the page.
    Acknowledged,
    /// Actively being investigated.
    Investigating,
    /// Short-term mitigation applied; user impact reduced but not eliminated.
    Mitigated,
    /// Service fully restored; incident resolved.
    Resolved,
    /// Post-mortem review in progress.
    PostMortem,
    /// Post-mortem complete; incident closed.
    Closed,
}

impl std::fmt::Display for IncidentLifecycle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Detected => "detected",
            Self::Triaged => "triaged",
            Self::Acknowledged => "acknowledged",
            Self::Investigating => "investigating",
            Self::Mitigated => "mitigated",
            Self::Resolved => "resolved",
            Self::PostMortem => "post-mortem",
            Self::Closed => "closed",
        };
        write!(f, "{s}")
    }
}

impl IncidentLifecycle {
    /// Returns `true` if the incident is still open (not yet resolved or closed).
    #[must_use]
    pub fn is_active(self) -> bool {
        !matches!(self, Self::Resolved | Self::PostMortem | Self::Closed)
    }

    /// Returns `true` if the incident has been resolved (Resolved, PostMortem, or Closed).
    #[must_use]
    pub fn is_resolved(self) -> bool {
        matches!(self, Self::Resolved | Self::PostMortem | Self::Closed)
    }
}

// ---------------------------------------------------------------------------
// Severity
// ---------------------------------------------------------------------------

/// Severity of a managed incident.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum IncidentSeverity {
    /// SEV-4: Minor — cosmetic or very limited user impact.
    Sev4,
    /// SEV-3: Moderate — partial feature degradation.
    Sev3,
    /// SEV-2: High — major feature unavailable.
    Sev2,
    /// SEV-1: Critical — complete outage or data loss risk.
    Sev1,
}

impl IncidentSeverity {
    /// Maximum escalation timeout (seconds) before paging the next on-call tier.
    #[must_use]
    pub fn escalation_timeout_secs(self) -> f64 {
        match self {
            Self::Sev4 => 3600.0,  // 1 h
            Self::Sev3 => 1800.0,  // 30 min
            Self::Sev2 => 900.0,   // 15 min
            Self::Sev1 => 300.0,   // 5 min
        }
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Sev4 => "SEV-4",
            Self::Sev3 => "SEV-3",
            Self::Sev2 => "SEV-2",
            Self::Sev1 => "SEV-1",
        }
    }
}

impl std::fmt::Display for IncidentSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ---------------------------------------------------------------------------
// Escalation Chain
// ---------------------------------------------------------------------------

/// A single escalation tier in an on-call chain.
#[derive(Debug, Clone)]
pub struct EscalationTier {
    /// Contact name or on-call alias.
    pub contact: String,
    /// Seconds after detection before this tier is paged.
    pub after_secs: f64,
    /// Whether this tier has been notified.
    pub notified: bool,
}

impl EscalationTier {
    /// Create a new tier.
    #[must_use]
    pub fn new(contact: impl Into<String>, after_secs: f64) -> Self {
        Self {
            contact: contact.into(),
            after_secs,
            notified: false,
        }
    }
}

/// Ordered escalation chain for an incident.
#[derive(Debug, Clone, Default)]
pub struct EscalationChain {
    tiers: Vec<EscalationTier>,
}

impl EscalationChain {
    /// Create an empty chain.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a tier (sorted by `after_secs` ascending).
    pub fn add_tier(&mut self, tier: EscalationTier) {
        self.tiers.push(tier);
        self.tiers
            .sort_by(|a, b| a.after_secs.partial_cmp(&b.after_secs).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Return tiers that should be notified given `elapsed_secs` since
    /// detection (and haven't been notified yet).
    pub fn due_notifications(&mut self, elapsed_secs: f64) -> Vec<&str> {
        let mut contacts = Vec::new();
        for tier in &mut self.tiers {
            if !tier.notified && elapsed_secs >= tier.after_secs {
                tier.notified = true;
                contacts.push(tier.contact.as_str());
            }
        }
        contacts
    }

    /// Number of tiers.
    #[must_use]
    pub fn tier_count(&self) -> usize {
        self.tiers.len()
    }

    /// Number of notified tiers.
    #[must_use]
    pub fn notified_count(&self) -> usize {
        self.tiers.iter().filter(|t| t.notified).count()
    }
}

// ---------------------------------------------------------------------------
// Post-mortem
// ---------------------------------------------------------------------------

/// Structured post-mortem document attached to a resolved incident.
#[derive(Debug, Clone)]
pub struct PostMortem {
    /// Root cause analysis (free text).
    pub root_cause: String,
    /// Ordered list of contributing factors.
    pub contributing_factors: Vec<String>,
    /// Action items (each is a description + optional owner).
    pub action_items: Vec<(String, Option<String>)>,
    /// Lessons learned.
    pub lessons_learned: String,
    /// Author of the post-mortem.
    pub author: String,
    /// Timestamp when the post-mortem was written (seconds since epoch).
    pub written_at_secs: f64,
}

impl PostMortem {
    /// Create a minimal post-mortem.
    #[must_use]
    pub fn new(
        root_cause: impl Into<String>,
        author: impl Into<String>,
        written_at_secs: f64,
    ) -> Self {
        Self {
            root_cause: root_cause.into(),
            contributing_factors: Vec::new(),
            action_items: Vec::new(),
            lessons_learned: String::new(),
            author: author.into(),
            written_at_secs,
        }
    }

    /// Add a contributing factor.
    pub fn add_factor(&mut self, factor: impl Into<String>) {
        self.contributing_factors.push(factor.into());
    }

    /// Add an action item with an optional owner.
    pub fn add_action_item(&mut self, description: impl Into<String>, owner: Option<&str>) {
        self.action_items
            .push((description.into(), owner.map(str::to_string)));
    }
}

// ---------------------------------------------------------------------------
// Timeline Entry
// ---------------------------------------------------------------------------

/// A single entry in an incident's timeline.
#[derive(Debug, Clone)]
pub struct TimelineEntry {
    /// Timestamp (seconds since epoch).
    pub timestamp_secs: f64,
    /// Author of this entry.
    pub author: String,
    /// Event description.
    pub description: String,
}

// ---------------------------------------------------------------------------
// Managed Incident
// ---------------------------------------------------------------------------

/// A fully managed incident record.
#[derive(Debug, Clone)]
pub struct ManagedIncident {
    /// Unique identifier.
    pub id: u64,
    /// Short title.
    pub title: String,
    /// Detailed description.
    pub description: String,
    /// Severity.
    pub severity: IncidentSeverity,
    /// Current lifecycle state.
    pub lifecycle: IncidentLifecycle,
    /// Timestamp when the underlying event actually started (seconds).
    pub event_start_secs: f64,
    /// Timestamp when the incident was first detected/created (seconds).
    pub detected_at_secs: f64,
    /// Timestamp when the incident was resolved, if applicable.
    pub resolved_at_secs: Option<f64>,
    /// User-impact score at open time (0 = no impact, 100 = complete outage).
    pub sli_impact_score: u8,
    /// On-call engineer currently responsible.
    pub assignee: Option<String>,
    /// Ordered escalation chain.
    pub escalation_chain: EscalationChain,
    /// Chronological timeline of events.
    pub timeline: Vec<TimelineEntry>,
    /// Post-mortem, once authored.
    pub post_mortem: Option<PostMortem>,
    /// Free-form tags.
    pub tags: Vec<String>,
}

impl ManagedIncident {
    /// Construct a new incident.
    #[must_use]
    pub fn new(
        id: u64,
        title: impl Into<String>,
        description: impl Into<String>,
        severity: IncidentSeverity,
        event_start_secs: f64,
        detected_at_secs: f64,
    ) -> Self {
        Self {
            id,
            title: title.into(),
            description: description.into(),
            severity,
            lifecycle: IncidentLifecycle::Detected,
            event_start_secs,
            detected_at_secs,
            resolved_at_secs: None,
            sli_impact_score: 0,
            assignee: None,
            escalation_chain: EscalationChain::new(),
            timeline: Vec::new(),
            post_mortem: None,
            tags: Vec::new(),
        }
    }

    /// Mean Time to Detect: duration from `event_start_secs` to `detected_at_secs`.
    #[must_use]
    pub fn mttd_secs(&self) -> f64 {
        (self.detected_at_secs - self.event_start_secs).max(0.0)
    }

    /// Mean Time to Recover: duration from `detected_at_secs` to resolution.
    ///
    /// Returns `None` if the incident is not yet resolved.
    #[must_use]
    pub fn mttr_secs(&self) -> Option<f64> {
        self.resolved_at_secs
            .map(|r| (r - self.detected_at_secs).max(0.0))
    }

    /// Total incident duration from `event_start_secs` to resolution (or `now`).
    #[must_use]
    pub fn total_duration_secs(&self, now: f64) -> f64 {
        let end = self.resolved_at_secs.unwrap_or(now);
        (end - self.event_start_secs).max(0.0)
    }

    /// Transition to a new lifecycle state, appending a timeline entry.
    pub fn transition(
        &mut self,
        new_state: IncidentLifecycle,
        author: impl Into<String>,
        timestamp_secs: f64,
        note: impl Into<String>,
    ) {
        self.lifecycle = new_state;
        if new_state.is_resolved() && self.resolved_at_secs.is_none() {
            self.resolved_at_secs = Some(timestamp_secs);
        }
        self.timeline.push(TimelineEntry {
            timestamp_secs,
            author: author.into(),
            description: note.into(),
        });
    }

    /// Assign to an engineer.
    pub fn assign(&mut self, assignee: impl Into<String>, timestamp_secs: f64, author: impl Into<String>) {
        let name: String = assignee.into();
        let entry = format!("Assigned to {name}");
        self.assignee = Some(name);
        self.timeline.push(TimelineEntry {
            timestamp_secs,
            author: author.into(),
            description: entry,
        });
    }

    /// Update the SLI impact score (0–100).
    pub fn set_impact(&mut self, score: u8) {
        self.sli_impact_score = score.min(100);
    }

    /// Attach a post-mortem document and transition to PostMortem state.
    pub fn attach_post_mortem(&mut self, pm: PostMortem) {
        let ts = pm.written_at_secs;
        let author = pm.author.clone();
        self.post_mortem = Some(pm);
        self.transition(
            IncidentLifecycle::PostMortem,
            author,
            ts,
            "Post-mortem attached",
        );
    }

    /// Add a tag.
    pub fn add_tag(&mut self, tag: impl Into<String>) {
        let tag = tag.into();
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }

    /// Check whether escalation should proceed given `elapsed_secs` since detection.
    ///
    /// Returns the list of newly-notified contacts.
    pub fn check_escalation(&mut self, elapsed_secs: f64) -> Vec<String> {
        self.escalation_chain
            .due_notifications(elapsed_secs)
            .into_iter()
            .map(str::to_string)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Incident Manager
// ---------------------------------------------------------------------------

/// Aggregate statistics produced by [`IncidentManager::statistics`].
#[derive(Debug, Clone)]
pub struct IncidentStats {
    /// Total number of incidents (open + closed).
    pub total: usize,
    /// Number of currently active incidents.
    pub active: usize,
    /// Number of resolved incidents.
    pub resolved: usize,
    /// Mean Time to Detect across all incidents (seconds).
    pub mean_mttd_secs: Option<f64>,
    /// Mean Time to Recover across resolved incidents (seconds).
    pub mean_mttr_secs: Option<f64>,
    /// Average SLI impact score across all incidents.
    pub avg_sli_impact: f64,
    /// Per-severity breakdown (severity → count).
    pub by_severity: HashMap<IncidentSeverity, usize>,
}

/// Full incident manager: creates, updates, and queries [`ManagedIncident`] records.
#[derive(Debug, Default)]
pub struct IncidentManager {
    incidents: HashMap<u64, ManagedIncident>,
    next_id: u64,
}

impl IncidentManager {
    /// Create a new, empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            incidents: HashMap::new(),
            next_id: 1,
        }
    }

    /// Open a new incident and return its ID.
    pub fn open(
        &mut self,
        title: impl Into<String>,
        description: impl Into<String>,
        severity: IncidentSeverity,
        event_start_secs: f64,
        detected_at_secs: f64,
        sli_impact: u8,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let mut inc = ManagedIncident::new(
            id,
            title,
            description,
            severity,
            event_start_secs,
            detected_at_secs,
        );
        inc.set_impact(sli_impact);
        self.incidents.insert(id, inc);
        id
    }

    /// Get an immutable reference to an incident.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&ManagedIncident> {
        self.incidents.get(&id)
    }

    /// Get a mutable reference to an incident.
    pub fn get_mut(&mut self, id: u64) -> Option<&mut ManagedIncident> {
        self.incidents.get_mut(&id)
    }

    /// Transition an incident's lifecycle state.
    ///
    /// Returns `false` if the incident does not exist.
    pub fn transition(
        &mut self,
        id: u64,
        state: IncidentLifecycle,
        author: impl Into<String>,
        timestamp_secs: f64,
        note: impl Into<String>,
    ) -> bool {
        if let Some(inc) = self.incidents.get_mut(&id) {
            inc.transition(state, author, timestamp_secs, note);
            true
        } else {
            false
        }
    }

    /// Resolve an incident.
    pub fn resolve(
        &mut self,
        id: u64,
        author: impl Into<String>,
        timestamp_secs: f64,
        note: impl Into<String>,
    ) -> bool {
        self.transition(id, IncidentLifecycle::Resolved, author, timestamp_secs, note)
    }

    /// Return all active incidents (sorted by severity descending).
    #[must_use]
    pub fn active_incidents(&self) -> Vec<&ManagedIncident> {
        let mut active: Vec<&ManagedIncident> = self
            .incidents
            .values()
            .filter(|i| i.lifecycle.is_active())
            .collect();
        active.sort_by(|a, b| b.severity.cmp(&a.severity));
        active
    }

    /// Return all resolved incidents.
    #[must_use]
    pub fn resolved_incidents(&self) -> Vec<&ManagedIncident> {
        self.incidents
            .values()
            .filter(|i| i.lifecycle.is_resolved())
            .collect()
    }

    /// Return incidents with a specific severity.
    #[must_use]
    pub fn by_severity(&self, severity: IncidentSeverity) -> Vec<&ManagedIncident> {
        self.incidents
            .values()
            .filter(|i| i.severity == severity)
            .collect()
    }

    /// Return incidents tagged with the given tag.
    #[must_use]
    pub fn by_tag(&self, tag: &str) -> Vec<&ManagedIncident> {
        self.incidents
            .values()
            .filter(|i| i.tags.iter().any(|t| t == tag))
            .collect()
    }

    /// Compute aggregate statistics.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn statistics(&self) -> IncidentStats {
        let total = self.incidents.len();
        let active = self.incidents.values().filter(|i| i.lifecycle.is_active()).count();
        let resolved = self.incidents.values().filter(|i| i.lifecycle.is_resolved()).count();

        // MTTD — all incidents.
        let mttd_values: Vec<f64> = self.incidents.values().map(|i| i.mttd_secs()).collect();
        let mean_mttd_secs = if mttd_values.is_empty() {
            None
        } else {
            Some(mttd_values.iter().sum::<f64>() / mttd_values.len() as f64)
        };

        // MTTR — resolved only.
        let mttr_values: Vec<f64> = self
            .incidents
            .values()
            .filter_map(|i| i.mttr_secs())
            .collect();
        let mean_mttr_secs = if mttr_values.is_empty() {
            None
        } else {
            Some(mttr_values.iter().sum::<f64>() / mttr_values.len() as f64)
        };

        // Average SLI impact.
        let avg_sli_impact = if total == 0 {
            0.0
        } else {
            let sum: f64 = self
                .incidents
                .values()
                .map(|i| f64::from(i.sli_impact_score))
                .sum();
            sum / total as f64
        };

        // Per-severity counts.
        let mut by_severity: HashMap<IncidentSeverity, usize> = HashMap::new();
        for inc in self.incidents.values() {
            *by_severity.entry(inc.severity).or_insert(0) += 1;
        }

        IncidentStats {
            total,
            active,
            resolved,
            mean_mttd_secs,
            mean_mttr_secs,
            avg_sli_impact,
            by_severity,
        }
    }

    /// Total number of incidents (active + resolved).
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.incidents.len()
    }

    /// Purge incidents that were resolved/closed before `cutoff_secs`.
    pub fn purge_before(&mut self, cutoff_secs: f64) -> usize {
        let before = self.incidents.len();
        self.incidents.retain(|_, i| {
            i.lifecycle.is_active()
                || i.resolved_at_secs.map_or(true, |t| t >= cutoff_secs)
        });
        before - self.incidents.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn open_incident(mgr: &mut IncidentManager) -> u64 {
        mgr.open(
            "Encoding pipeline stalled",
            "VP9 encoder hung on worker-03",
            IncidentSeverity::Sev2,
            1000.0, // event_start
            1060.0, // detected_at (MTTD = 60 s)
            80,     // SLI impact
        )
    }

    #[test]
    fn test_lifecycle_is_active() {
        assert!(IncidentLifecycle::Detected.is_active());
        assert!(IncidentLifecycle::Investigating.is_active());
        assert!(!IncidentLifecycle::Resolved.is_active());
        assert!(!IncidentLifecycle::Closed.is_active());
    }

    #[test]
    fn test_lifecycle_display() {
        assert_eq!(IncidentLifecycle::PostMortem.to_string(), "post-mortem");
        assert_eq!(IncidentLifecycle::Mitigated.to_string(), "mitigated");
    }

    #[test]
    fn test_severity_ordering() {
        assert!(IncidentSeverity::Sev1 > IncidentSeverity::Sev4);
    }

    #[test]
    fn test_severity_escalation_timeout() {
        assert!(
            IncidentSeverity::Sev1.escalation_timeout_secs()
                < IncidentSeverity::Sev4.escalation_timeout_secs()
        );
    }

    #[test]
    fn test_manager_open_and_get() {
        let mut mgr = IncidentManager::new();
        let id = open_incident(&mut mgr);
        let inc = mgr.get(id).expect("incident should exist");
        assert_eq!(inc.severity, IncidentSeverity::Sev2);
        assert_eq!(inc.lifecycle, IncidentLifecycle::Detected);
        assert_eq!(inc.sli_impact_score, 80);
    }

    #[test]
    fn test_mttd_calculation() {
        let mut mgr = IncidentManager::new();
        let id = open_incident(&mut mgr);
        let inc = mgr.get(id).expect("incident should exist");
        // event_start=1000, detected_at=1060 → MTTD = 60 s
        assert!((inc.mttd_secs() - 60.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mttr_calculation() {
        let mut mgr = IncidentManager::new();
        let id = open_incident(&mut mgr);
        mgr.resolve(id, "alice", 1660.0, "Restarted encoder");
        let inc = mgr.get(id).expect("incident should exist");
        // detected_at=1060, resolved_at=1660 → MTTR = 600 s
        let mttr = inc.mttr_secs().expect("should have MTTR");
        assert!((mttr - 600.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_transition_changes_lifecycle() {
        let mut mgr = IncidentManager::new();
        let id = open_incident(&mut mgr);
        mgr.transition(
            id,
            IncidentLifecycle::Investigating,
            "bob",
            1100.0,
            "Assigned to on-call",
        );
        let inc = mgr.get(id).expect("incident should exist");
        assert_eq!(inc.lifecycle, IncidentLifecycle::Investigating);
        assert_eq!(inc.timeline.len(), 1);
    }

    #[test]
    fn test_active_incidents_sorted_by_severity() {
        let mut mgr = IncidentManager::new();
        mgr.open("low", "", IncidentSeverity::Sev4, 0.0, 1.0, 10);
        mgr.open("high", "", IncidentSeverity::Sev1, 0.0, 1.0, 90);
        mgr.open("mid", "", IncidentSeverity::Sev2, 0.0, 1.0, 50);
        let active = mgr.active_incidents();
        assert_eq!(active.len(), 3);
        // First should be SEV-1 (highest).
        assert_eq!(active[0].severity, IncidentSeverity::Sev1);
    }

    #[test]
    fn test_resolve_moves_to_resolved() {
        let mut mgr = IncidentManager::new();
        let id = open_incident(&mut mgr);
        mgr.resolve(id, "carol", 2000.0, "All clear");
        let inc = mgr.get(id).expect("incident should exist");
        assert!(inc.lifecycle.is_resolved());
        assert_eq!(mgr.resolved_incidents().len(), 1);
        assert_eq!(mgr.active_incidents().len(), 0);
    }

    #[test]
    fn test_statistics_mttd_mttr() {
        let mut mgr = IncidentManager::new();
        let id1 = mgr.open("A", "", IncidentSeverity::Sev2, 1000.0, 1100.0, 50); // MTTD=100
        let id2 = mgr.open("B", "", IncidentSeverity::Sev3, 2000.0, 2050.0, 30); // MTTD=50
        mgr.resolve(id1, "x", 1700.0, "done"); // MTTR=600
        mgr.resolve(id2, "y", 2350.0, "done"); // MTTR=300
        let stats = mgr.statistics();
        let mttd = stats.mean_mttd_secs.expect("mttd should exist");
        assert!((mttd - 75.0).abs() < f64::EPSILON); // (100+50)/2
        let mttr = stats.mean_mttr_secs.expect("mttr should exist");
        assert!((mttr - 450.0).abs() < f64::EPSILON); // (600+300)/2
    }

    #[test]
    fn test_escalation_chain() {
        let mut chain = EscalationChain::new();
        chain.add_tier(EscalationTier::new("primary", 0.0));
        chain.add_tier(EscalationTier::new("secondary", 300.0));
        chain.add_tier(EscalationTier::new("manager", 600.0));
        let notified = chain.due_notifications(350.0);
        assert_eq!(notified.len(), 2);
        assert_eq!(notified[0], "primary");
        assert_eq!(notified[1], "secondary");
        assert_eq!(chain.notified_count(), 2);
    }

    #[test]
    fn test_post_mortem_attach() {
        let mut mgr = IncidentManager::new();
        let id = open_incident(&mut mgr);
        mgr.resolve(id, "alice", 2000.0, "resolved");

        let mut pm = PostMortem::new("Memory leak in VP9 encoder", "alice", 2100.0);
        pm.add_factor("Missing buffer cleanup on frame drop");
        pm.add_action_item("Add cleanup in encoder teardown", Some("bob"));

        let inc = mgr.get_mut(id).expect("incident should exist");
        inc.attach_post_mortem(pm);

        let inc = mgr.get(id).expect("incident should exist");
        assert!(inc.post_mortem.is_some());
        assert_eq!(inc.lifecycle, IncidentLifecycle::PostMortem);
    }

    #[test]
    fn test_purge_before() {
        let mut mgr = IncidentManager::new();
        let id1 = mgr.open("old", "", IncidentSeverity::Sev4, 100.0, 101.0, 5);
        mgr.open("new", "", IncidentSeverity::Sev4, 200.0, 201.0, 5);
        mgr.open("active", "", IncidentSeverity::Sev2, 300.0, 301.0, 50);
        mgr.resolve(id1, "x", 150.0, "done");
        // id2 resolved after cutoff.
        let id2 = mgr.open("mid", "", IncidentSeverity::Sev4, 200.0, 201.0, 5);
        mgr.resolve(id2, "y", 250.0, "done");

        let purged = mgr.purge_before(200.0);
        // id1 resolved at 150 < 200 → purged
        assert_eq!(purged, 1);
    }

    #[test]
    fn test_by_tag_filtering() {
        let mut mgr = IncidentManager::new();
        let id = open_incident(&mut mgr);
        mgr.get_mut(id).expect("exists").add_tag("video-pipeline");
        mgr.open("unrelated", "", IncidentSeverity::Sev4, 0.0, 0.0, 0);

        let tagged = mgr.by_tag("video-pipeline");
        assert_eq!(tagged.len(), 1);
        assert_eq!(tagged[0].id, id);
    }

    #[test]
    fn test_total_duration() {
        let inc = ManagedIncident::new(
            1, "test", "desc",
            IncidentSeverity::Sev3,
            1000.0,
            1100.0,
        );
        // Not yet resolved — duration from event_start to "now".
        assert!((inc.total_duration_secs(1600.0) - 600.0).abs() < f64::EPSILON);
    }
}
