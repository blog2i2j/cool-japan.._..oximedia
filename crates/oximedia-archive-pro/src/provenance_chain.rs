#![allow(dead_code)]
//! Provenance chain tracking for archived digital assets.
//!
//! This module records the complete chain of custody for every archived item,
//! from initial ingest through every transformation, migration, and access event.
//! It supports verifiable provenance records with cryptographic linking.

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Type of provenance event recorded in the chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProvenanceEventKind {
    /// Asset was ingested into the archive.
    Ingest,
    /// Asset was accessed (read).
    Access,
    /// Asset underwent format migration.
    Migration,
    /// Asset metadata was updated.
    MetadataUpdate,
    /// Asset was replicated to another location.
    Replication,
    /// Fixity check was performed.
    FixityCheck,
    /// Asset was restored from archive.
    Restore,
    /// Asset was transferred to another custodian.
    CustodyTransfer,
    /// Asset was quarantined due to integrity issues.
    Quarantine,
    /// Asset was released from quarantine.
    QuarantineRelease,
}

impl ProvenanceEventKind {
    /// Returns a human-readable label for this event kind.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Ingest => "Ingest",
            Self::Access => "Access",
            Self::Migration => "Migration",
            Self::MetadataUpdate => "Metadata Update",
            Self::Replication => "Replication",
            Self::FixityCheck => "Fixity Check",
            Self::Restore => "Restore",
            Self::CustodyTransfer => "Custody Transfer",
            Self::Quarantine => "Quarantine",
            Self::QuarantineRelease => "Quarantine Release",
        }
    }

    /// Returns `true` if this event modifies the asset content.
    #[must_use]
    pub const fn modifies_content(&self) -> bool {
        matches!(self, Self::Migration)
    }
}

/// A single provenance event in the chain.
#[derive(Debug, Clone)]
pub struct ProvenanceEvent {
    /// Unique identifier for this event.
    pub event_id: u64,
    /// The kind of event.
    pub kind: ProvenanceEventKind,
    /// Timestamp when the event occurred.
    pub timestamp: SystemTime,
    /// Agent (user or system) that performed the action.
    pub agent: String,
    /// Description of the event.
    pub description: String,
    /// Optional hash of the asset state after this event.
    pub state_hash: Option<String>,
    /// Hash of the previous event for chain integrity.
    pub previous_hash: Option<String>,
    /// Additional key-value metadata.
    pub metadata: HashMap<String, String>,
}

impl ProvenanceEvent {
    /// Creates a new provenance event.
    #[must_use]
    pub fn new(event_id: u64, kind: ProvenanceEventKind, agent: &str, description: &str) -> Self {
        Self {
            event_id,
            kind,
            timestamp: SystemTime::now(),
            agent: agent.to_string(),
            description: description.to_string(),
            state_hash: None,
            previous_hash: None,
            metadata: HashMap::new(),
        }
    }

    /// Sets the state hash after this event.
    #[must_use]
    pub fn with_state_hash(mut self, hash: &str) -> Self {
        self.state_hash = Some(hash.to_string());
        self
    }

    /// Adds a metadata entry.
    #[must_use]
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Computes a simple hash of this event for chaining.
    #[must_use]
    pub fn compute_chain_hash(&self) -> String {
        let mut hasher_state: u64 = 0xcbf2_9ce4_8422_2325;
        let data = format!(
            "{}:{}:{}:{}",
            self.event_id,
            self.kind.label(),
            self.agent,
            self.description,
        );
        for byte in data.as_bytes() {
            hasher_state ^= u64::from(*byte);
            hasher_state = hasher_state.wrapping_mul(0x0100_0000_01b3);
        }
        format!("{hasher_state:016x}")
    }
}

/// Integrity status of the provenance chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainIntegrity {
    /// Chain is valid and unbroken.
    Valid,
    /// Chain has a broken link at the given event index.
    BrokenAt(usize),
    /// Chain is empty.
    Empty,
}

/// The complete provenance chain for an archived asset.
#[derive(Debug, Clone)]
pub struct ProvenanceChain {
    /// Asset identifier.
    pub asset_id: String,
    /// Ordered list of provenance events.
    events: Vec<ProvenanceEvent>,
    /// Next event ID to assign.
    next_id: u64,
}

impl ProvenanceChain {
    /// Creates a new empty provenance chain for an asset.
    #[must_use]
    pub fn new(asset_id: &str) -> Self {
        Self {
            asset_id: asset_id.to_string(),
            events: Vec::new(),
            next_id: 1,
        }
    }

    /// Appends a new event to the chain, linking it to the previous event.
    pub fn append_event(
        &mut self,
        kind: ProvenanceEventKind,
        agent: &str,
        description: &str,
    ) -> &ProvenanceEvent {
        let previous_hash = self.events.last().map(|e| e.compute_chain_hash());
        let mut event = ProvenanceEvent::new(self.next_id, kind, agent, description);
        event.previous_hash = previous_hash;
        self.next_id += 1;
        self.events.push(event);
        self.events.last().expect("just pushed an event")
    }

    /// Returns the number of events in the chain.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns `true` if the chain has no events.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Returns a reference to all events in the chain.
    #[must_use]
    pub fn events(&self) -> &[ProvenanceEvent] {
        &self.events
    }

    /// Returns events of a specific kind.
    #[must_use]
    pub fn events_of_kind(&self, kind: ProvenanceEventKind) -> Vec<&ProvenanceEvent> {
        self.events.iter().filter(|e| e.kind == kind).collect()
    }

    /// Returns the most recent event, if any.
    #[must_use]
    pub fn latest_event(&self) -> Option<&ProvenanceEvent> {
        self.events.last()
    }

    /// Verifies the integrity of the chain by checking that each event's
    /// `previous_hash` matches the computed hash of its predecessor.
    #[must_use]
    pub fn verify_integrity(&self) -> ChainIntegrity {
        if self.events.is_empty() {
            return ChainIntegrity::Empty;
        }
        // First event should have no previous hash
        if self.events[0].previous_hash.is_some() {
            return ChainIntegrity::BrokenAt(0);
        }
        for i in 1..self.events.len() {
            let expected = self.events[i - 1].compute_chain_hash();
            match &self.events[i].previous_hash {
                Some(prev) if prev == &expected => {}
                _ => return ChainIntegrity::BrokenAt(i),
            }
        }
        ChainIntegrity::Valid
    }

    /// Returns the total duration from first to last event.
    #[must_use]
    pub fn chain_duration(&self) -> Option<Duration> {
        if self.events.len() < 2 {
            return None;
        }
        let first = self.events.first()?.timestamp;
        let last = self.events.last()?.timestamp;
        last.duration_since(first).ok()
    }

    /// Counts events grouped by kind.
    #[must_use]
    pub fn event_counts(&self) -> HashMap<ProvenanceEventKind, usize> {
        let mut counts = HashMap::new();
        for event in &self.events {
            *counts.entry(event.kind).or_insert(0) += 1;
        }
        counts
    }

    /// Returns the last custodian agent in the chain.
    #[must_use]
    pub fn current_custodian(&self) -> Option<&str> {
        self.events
            .iter()
            .rev()
            .find(|e| {
                e.kind == ProvenanceEventKind::CustodyTransfer
                    || e.kind == ProvenanceEventKind::Ingest
            })
            .map(|e| e.agent.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_kind_label() {
        assert_eq!(ProvenanceEventKind::Ingest.label(), "Ingest");
        assert_eq!(ProvenanceEventKind::Migration.label(), "Migration");
        assert_eq!(ProvenanceEventKind::FixityCheck.label(), "Fixity Check");
    }

    #[test]
    fn test_event_kind_modifies_content() {
        assert!(ProvenanceEventKind::Migration.modifies_content());
        assert!(!ProvenanceEventKind::Access.modifies_content());
        assert!(!ProvenanceEventKind::Ingest.modifies_content());
    }

    #[test]
    fn test_new_chain_is_empty() {
        let chain = ProvenanceChain::new("asset-001");
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
        assert_eq!(chain.asset_id, "asset-001");
    }

    #[test]
    fn test_append_event() {
        let mut chain = ProvenanceChain::new("asset-002");
        chain.append_event(ProvenanceEventKind::Ingest, "archivist", "Initial ingest");
        assert_eq!(chain.len(), 1);
        assert!(!chain.is_empty());
    }

    #[test]
    fn test_chain_integrity_valid() {
        let mut chain = ProvenanceChain::new("asset-003");
        chain.append_event(ProvenanceEventKind::Ingest, "user1", "Ingested file");
        chain.append_event(
            ProvenanceEventKind::FixityCheck,
            "system",
            "Verified checksum",
        );
        chain.append_event(ProvenanceEventKind::Access, "user2", "Downloaded copy");
        assert_eq!(chain.verify_integrity(), ChainIntegrity::Valid);
    }

    #[test]
    fn test_chain_integrity_empty() {
        let chain = ProvenanceChain::new("asset-empty");
        assert_eq!(chain.verify_integrity(), ChainIntegrity::Empty);
    }

    #[test]
    fn test_chain_integrity_broken() {
        let mut chain = ProvenanceChain::new("asset-broken");
        chain.append_event(ProvenanceEventKind::Ingest, "user1", "Ingested");
        chain.append_event(ProvenanceEventKind::Access, "user2", "Accessed");
        // Tamper with the chain
        chain.events[1].previous_hash = Some("tampered".to_string());
        assert_eq!(chain.verify_integrity(), ChainIntegrity::BrokenAt(1));
    }

    #[test]
    fn test_events_of_kind() {
        let mut chain = ProvenanceChain::new("asset-004");
        chain.append_event(ProvenanceEventKind::Ingest, "u1", "Ingest 1");
        chain.append_event(ProvenanceEventKind::FixityCheck, "sys", "Check 1");
        chain.append_event(ProvenanceEventKind::FixityCheck, "sys", "Check 2");
        let fixity_events = chain.events_of_kind(ProvenanceEventKind::FixityCheck);
        assert_eq!(fixity_events.len(), 2);
    }

    #[test]
    fn test_latest_event() {
        let mut chain = ProvenanceChain::new("asset-005");
        assert!(chain.latest_event().is_none());
        chain.append_event(ProvenanceEventKind::Ingest, "u1", "First");
        chain.append_event(ProvenanceEventKind::Access, "u2", "Second");
        let latest = chain.latest_event().expect("operation should succeed");
        assert_eq!(latest.kind, ProvenanceEventKind::Access);
        assert_eq!(latest.description, "Second");
    }

    #[test]
    fn test_event_counts() {
        let mut chain = ProvenanceChain::new("asset-006");
        chain.append_event(ProvenanceEventKind::Ingest, "u1", "Ingest");
        chain.append_event(ProvenanceEventKind::Access, "u2", "Access 1");
        chain.append_event(ProvenanceEventKind::Access, "u3", "Access 2");
        chain.append_event(ProvenanceEventKind::Migration, "sys", "Migrate");
        let counts = chain.event_counts();
        assert_eq!(counts[&ProvenanceEventKind::Ingest], 1);
        assert_eq!(counts[&ProvenanceEventKind::Access], 2);
        assert_eq!(counts[&ProvenanceEventKind::Migration], 1);
    }

    #[test]
    fn test_current_custodian() {
        let mut chain = ProvenanceChain::new("asset-007");
        chain.append_event(ProvenanceEventKind::Ingest, "original_owner", "Ingest");
        chain.append_event(
            ProvenanceEventKind::CustodyTransfer,
            "new_owner",
            "Transfer",
        );
        assert_eq!(chain.current_custodian(), Some("new_owner"));
    }

    #[test]
    fn test_event_with_state_hash_and_metadata() {
        let event = ProvenanceEvent::new(1, ProvenanceEventKind::Ingest, "user1", "Test ingest")
            .with_state_hash("abc123")
            .with_metadata("source", "tape-42");
        assert_eq!(event.state_hash.as_deref(), Some("abc123"));
        assert_eq!(
            event.metadata.get("source").map(String::as_str),
            Some("tape-42")
        );
    }

    #[test]
    fn test_compute_chain_hash_deterministic() {
        let event = ProvenanceEvent::new(1, ProvenanceEventKind::Ingest, "user", "desc");
        let h1 = event.compute_chain_hash();
        let h2 = event.compute_chain_hash();
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 16);
    }
}
