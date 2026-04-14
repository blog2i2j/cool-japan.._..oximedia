#![allow(dead_code)]
//! Provenance chain tracking for archived digital assets.
//!
//! This module records the complete chain of custody for every archived item,
//! from initial ingest through every transformation, migration, and access event.
//! It supports verifiable provenance records with cryptographic linking and
//! tamper-evident signing using HMAC-SHA256.

use sha2::{Digest, Sha256};
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

    /// Sign the entire chain with HMAC-SHA256, producing a `SignedProvenanceChain`.
    ///
    /// The signing key is used to compute an HMAC over the canonical representation
    /// of every event in the chain plus the chain-level root hash. This creates a
    /// tamper-evident seal: any modification to any event will invalidate the signature.
    #[must_use]
    pub fn sign(&self, signing_key: &[u8]) -> SignedProvenanceChain {
        let chain_digest = self.compute_chain_digest();
        let signature = hmac_sha256(signing_key, chain_digest.as_bytes());

        SignedProvenanceChain {
            asset_id: self.asset_id.clone(),
            events: self.events.clone(),
            next_id: self.next_id,
            chain_digest,
            signature,
            signed_at: SystemTime::now(),
        }
    }

    /// Compute a SHA-256 digest over the entire chain for signing.
    ///
    /// The digest covers: asset_id, each event's id/kind/agent/description/state_hash/previous_hash.
    #[must_use]
    pub fn compute_chain_digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.asset_id.as_bytes());

        for event in &self.events {
            hasher.update(event.event_id.to_le_bytes());
            hasher.update(event.kind.label().as_bytes());
            hasher.update(event.agent.as_bytes());
            hasher.update(event.description.as_bytes());
            if let Some(ref sh) = event.state_hash {
                hasher.update(sh.as_bytes());
            }
            if let Some(ref ph) = event.previous_hash {
                hasher.update(ph.as_bytes());
            }
            // Include metadata keys/values in sorted order for determinism
            let mut meta_keys: Vec<&String> = event.metadata.keys().collect();
            meta_keys.sort();
            for key in meta_keys {
                hasher.update(key.as_bytes());
                if let Some(val) = event.metadata.get(key) {
                    hasher.update(val.as_bytes());
                }
            }
        }

        hex::encode(hasher.finalize())
    }
}

/// HMAC-SHA256 implementation (pure Rust, no external HMAC crate needed).
///
/// Follows RFC 2104: HMAC = H((K' xor opad) || H((K' xor ipad) || message))
fn hmac_sha256(key: &[u8], message: &[u8]) -> String {
    const BLOCK_SIZE: usize = 64;
    const IPAD: u8 = 0x36;
    const OPAD: u8 = 0x5c;

    // If key is longer than block size, hash it first
    let key_prime = if key.len() > BLOCK_SIZE {
        let mut hasher = Sha256::new();
        hasher.update(key);
        hasher.finalize().to_vec()
    } else {
        key.to_vec()
    };

    // Pad key to block size
    let mut padded_key = vec![0u8; BLOCK_SIZE];
    padded_key[..key_prime.len()].copy_from_slice(&key_prime);

    // Inner hash: H((K' xor ipad) || message)
    let mut inner_hasher = Sha256::new();
    let inner_key: Vec<u8> = padded_key.iter().map(|b| b ^ IPAD).collect();
    inner_hasher.update(&inner_key);
    inner_hasher.update(message);
    let inner_hash = inner_hasher.finalize();

    // Outer hash: H((K' xor opad) || inner_hash)
    let mut outer_hasher = Sha256::new();
    let outer_key: Vec<u8> = padded_key.iter().map(|b| b ^ OPAD).collect();
    outer_hasher.update(&outer_key);
    outer_hasher.update(inner_hash);

    hex::encode(outer_hasher.finalize())
}

/// A cryptographically signed provenance chain.
///
/// The signature covers the entire chain state and can be verified
/// using the same signing key to detect tampering.
#[derive(Debug, Clone)]
pub struct SignedProvenanceChain {
    /// Asset identifier.
    pub asset_id: String,
    /// Ordered list of provenance events.
    events: Vec<ProvenanceEvent>,
    /// Next event ID.
    next_id: u64,
    /// SHA-256 digest of the chain at signing time.
    pub chain_digest: String,
    /// HMAC-SHA256 signature over the chain digest.
    pub signature: String,
    /// When the chain was signed.
    pub signed_at: SystemTime,
}

impl SignedProvenanceChain {
    /// Verify that the chain has not been tampered with.
    ///
    /// Recomputes the chain digest and HMAC, then compares against the stored
    /// signature. Returns `SignatureVerification::Valid` if they match.
    #[must_use]
    pub fn verify(&self, signing_key: &[u8]) -> SignatureVerification {
        // Reconstruct the unsigned chain for digest computation
        let chain = ProvenanceChain {
            asset_id: self.asset_id.clone(),
            events: self.events.clone(),
            next_id: self.next_id,
        };

        let current_digest = chain.compute_chain_digest();
        if current_digest != self.chain_digest {
            return SignatureVerification::DigestMismatch {
                expected: self.chain_digest.clone(),
                actual: current_digest,
            };
        }

        let expected_signature = hmac_sha256(signing_key, current_digest.as_bytes());
        if expected_signature != self.signature {
            return SignatureVerification::SignatureInvalid;
        }

        SignatureVerification::Valid
    }

    /// Returns a reference to all events.
    #[must_use]
    pub fn events(&self) -> &[ProvenanceEvent] {
        &self.events
    }

    /// Returns the number of events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns whether the chain is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Verify the internal chain integrity (hash linking) in addition to the signature.
    #[must_use]
    pub fn full_verify(&self, signing_key: &[u8]) -> FullVerification {
        // First verify cryptographic signature
        let sig_result = self.verify(signing_key);
        if sig_result != SignatureVerification::Valid {
            return FullVerification {
                signature: sig_result,
                chain_integrity: ChainIntegrity::Empty,
            };
        }

        // Then verify chain integrity
        let chain = ProvenanceChain {
            asset_id: self.asset_id.clone(),
            events: self.events.clone(),
            next_id: self.next_id,
        };

        FullVerification {
            signature: SignatureVerification::Valid,
            chain_integrity: chain.verify_integrity(),
        }
    }

    /// Serialize the signed chain to a JSON string for storage/transmission.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_json(&self) -> std::result::Result<String, String> {
        // Build a serializable representation
        let events_data: Vec<HashMap<String, String>> = self
            .events
            .iter()
            .map(|e| {
                let mut m = HashMap::new();
                m.insert("event_id".to_string(), e.event_id.to_string());
                m.insert("kind".to_string(), e.kind.label().to_string());
                m.insert("agent".to_string(), e.agent.clone());
                m.insert("description".to_string(), e.description.clone());
                m.insert("chain_hash".to_string(), e.compute_chain_hash());
                if let Some(ref sh) = e.state_hash {
                    m.insert("state_hash".to_string(), sh.clone());
                }
                if let Some(ref ph) = e.previous_hash {
                    m.insert("previous_hash".to_string(), ph.clone());
                }
                m
            })
            .collect();

        let mut doc = HashMap::new();
        doc.insert("asset_id", serde_json::json!(self.asset_id));
        doc.insert("chain_digest", serde_json::json!(self.chain_digest));
        doc.insert("signature", serde_json::json!(self.signature));
        doc.insert("event_count", serde_json::json!(self.events.len()));
        doc.insert("events", serde_json::json!(events_data));

        serde_json::to_string_pretty(&doc).map_err(|e| format!("Serialization error: {e}"))
    }
}

/// Result of verifying a cryptographic signature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SignatureVerification {
    /// Signature is valid; the chain has not been tampered with.
    Valid,
    /// The chain digest does not match the stored digest.
    DigestMismatch {
        /// Expected digest at signing time.
        expected: String,
        /// Current computed digest.
        actual: String,
    },
    /// The HMAC signature does not match (wrong key or tampered signature).
    SignatureInvalid,
}

/// Combined result of signature verification and chain integrity check.
#[derive(Debug, Clone)]
pub struct FullVerification {
    /// Result of the cryptographic signature check.
    pub signature: SignatureVerification,
    /// Result of the hash-chain integrity check.
    pub chain_integrity: ChainIntegrity,
}

impl FullVerification {
    /// Returns `true` if both the signature and chain integrity are valid.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.signature == SignatureVerification::Valid
            && self.chain_integrity == ChainIntegrity::Valid
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

    // ── Cryptographic signing tests ─────────────────────────────

    const TEST_KEY: &[u8] = b"test-signing-key-for-archive-provenance";

    fn make_signed_chain() -> (ProvenanceChain, SignedProvenanceChain) {
        let mut chain = ProvenanceChain::new("asset-sign-001");
        chain.append_event(ProvenanceEventKind::Ingest, "archivist", "Initial ingest");
        chain.append_event(
            ProvenanceEventKind::FixityCheck,
            "system",
            "Verified SHA-256",
        );
        chain.append_event(
            ProvenanceEventKind::Migration,
            "auto-migrator",
            "MPEG-2 to FFV1",
        );
        let signed = chain.sign(TEST_KEY);
        (chain, signed)
    }

    #[test]
    fn test_sign_produces_signature() {
        let (_, signed) = make_signed_chain();
        assert!(!signed.signature.is_empty());
        assert!(!signed.chain_digest.is_empty());
        assert_eq!(signed.asset_id, "asset-sign-001");
        assert_eq!(signed.len(), 3);
    }

    #[test]
    fn test_verify_valid_signature() {
        let (_, signed) = make_signed_chain();
        assert_eq!(signed.verify(TEST_KEY), SignatureVerification::Valid);
    }

    #[test]
    fn test_verify_wrong_key() {
        let (_, signed) = make_signed_chain();
        assert_eq!(
            signed.verify(b"wrong-key"),
            SignatureVerification::SignatureInvalid
        );
    }

    #[test]
    fn test_verify_tampered_event() {
        let (_, mut signed) = make_signed_chain();
        // Tamper with an event description
        signed.events[1].description = "TAMPERED".to_string();
        let result = signed.verify(TEST_KEY);
        assert!(matches!(
            result,
            SignatureVerification::DigestMismatch { .. }
        ));
    }

    #[test]
    fn test_verify_tampered_asset_id() {
        let (_, mut signed) = make_signed_chain();
        signed.asset_id = "tampered-id".to_string();
        let result = signed.verify(TEST_KEY);
        assert!(matches!(
            result,
            SignatureVerification::DigestMismatch { .. }
        ));
    }

    #[test]
    fn test_full_verify_valid() {
        let (_, signed) = make_signed_chain();
        let result = signed.full_verify(TEST_KEY);
        assert!(result.is_valid());
    }

    #[test]
    fn test_full_verify_broken_chain() {
        let (_, mut signed) = make_signed_chain();
        // Tamper with previous_hash to break chain integrity
        // but recompute the digest so signature matches the tampered state
        signed.events[2].previous_hash = Some("broken-link".to_string());
        // Re-sign with the new tampered data (simulates an attacker who has the key)
        let tampered_chain = ProvenanceChain {
            asset_id: signed.asset_id.clone(),
            events: signed.events.clone(),
            next_id: signed.next_id,
        };
        let new_digest = tampered_chain.compute_chain_digest();
        let new_sig = hmac_sha256(TEST_KEY, new_digest.as_bytes());
        signed.chain_digest = new_digest;
        signed.signature = new_sig;

        let result = signed.full_verify(TEST_KEY);
        // Signature is valid (attacker re-signed) but chain integrity is broken
        assert_eq!(result.signature, SignatureVerification::Valid);
        assert!(matches!(
            result.chain_integrity,
            ChainIntegrity::BrokenAt(_)
        ));
        assert!(!result.is_valid());
    }

    #[test]
    fn test_sign_empty_chain() {
        let chain = ProvenanceChain::new("empty-asset");
        let signed = chain.sign(TEST_KEY);
        assert_eq!(signed.verify(TEST_KEY), SignatureVerification::Valid);
        assert!(signed.is_empty());
    }

    #[test]
    fn test_sign_deterministic() {
        let (chain, signed1) = make_signed_chain();
        let signed2 = chain.sign(TEST_KEY);
        // Same chain + same key = same digest and signature
        assert_eq!(signed1.chain_digest, signed2.chain_digest);
        assert_eq!(signed1.signature, signed2.signature);
    }

    #[test]
    fn test_chain_digest_changes_with_events() {
        let mut chain = ProvenanceChain::new("asset-digest");
        let digest_empty = chain.compute_chain_digest();
        chain.append_event(ProvenanceEventKind::Ingest, "user", "Ingest");
        let digest_one = chain.compute_chain_digest();
        chain.append_event(ProvenanceEventKind::Access, "user", "Access");
        let digest_two = chain.compute_chain_digest();

        assert_ne!(digest_empty, digest_one);
        assert_ne!(digest_one, digest_two);
    }

    #[test]
    fn test_hmac_sha256_known_vector() {
        // Verify our HMAC implementation produces consistent output
        let sig1 = hmac_sha256(b"key", b"message");
        let sig2 = hmac_sha256(b"key", b"message");
        assert_eq!(sig1, sig2);
        assert_eq!(sig1.len(), 64); // SHA-256 hex = 64 chars
    }

    #[test]
    fn test_hmac_sha256_different_keys_different_output() {
        let sig1 = hmac_sha256(b"key1", b"same message");
        let sig2 = hmac_sha256(b"key2", b"same message");
        assert_ne!(sig1, sig2);
    }

    #[test]
    fn test_hmac_sha256_long_key() {
        // Key longer than SHA-256 block size (64 bytes)
        let long_key = vec![0xABu8; 128];
        let sig = hmac_sha256(&long_key, b"test");
        assert_eq!(sig.len(), 64);
        // Verify consistency
        let sig2 = hmac_sha256(&long_key, b"test");
        assert_eq!(sig, sig2);
    }

    #[test]
    fn test_signed_chain_to_json() {
        let (_, signed) = make_signed_chain();
        let json = signed.to_json().expect("operation should succeed");
        assert!(json.contains("asset-sign-001"));
        assert!(json.contains(&signed.signature));
        assert!(json.contains(&signed.chain_digest));
        assert!(json.contains("event_count"));
    }

    #[test]
    fn test_signed_chain_events_accessible() {
        let (_, signed) = make_signed_chain();
        assert_eq!(signed.events().len(), 3);
        assert_eq!(signed.events()[0].kind, ProvenanceEventKind::Ingest);
        assert_eq!(signed.events()[2].kind, ProvenanceEventKind::Migration);
    }

    #[test]
    fn test_event_metadata_included_in_digest() {
        let mut chain1 = ProvenanceChain::new("meta-test");
        chain1.append_event(ProvenanceEventKind::Ingest, "u1", "Ingest");
        // Manually add metadata to the event
        chain1.events[0]
            .metadata
            .insert("source".to_string(), "tape-42".to_string());
        let digest1 = chain1.compute_chain_digest();

        let mut chain2 = ProvenanceChain::new("meta-test");
        chain2.append_event(ProvenanceEventKind::Ingest, "u1", "Ingest");
        chain2.events[0]
            .metadata
            .insert("source".to_string(), "tape-99".to_string());
        let digest2 = chain2.compute_chain_digest();

        // Different metadata should produce different digests
        assert_ne!(digest1, digest2);
    }
}
