//! Chain of custody tracking for digital media assets.
//!
//! [`ChainOfCustody`] records a tamper-evident sequence of *custody events*,
//! each stamped with an actor identifier, an action description, and a
//! timestamp.  The chain is verified by checking that events are in non-
//! decreasing timestamp order and that each event's checksum references the
//! previous event's checksum (hash chaining).
//!
//! # Hash chaining
//!
//! Each event carries a 64-bit FNV-1a checksum computed over:
//! `prev_checksum || asset_id || sequence_number || actor || action || timestamp`
//!
//! The `verify()` method recomputes every checksum from scratch and confirms
//! that:
//! 1. The first event's previous-checksum field is `0` (genesis).
//! 2. Each subsequent event's previous-checksum equals the computed checksum
//!    of its predecessor.
//! 3. Timestamps are non-decreasing.
//!
//! # Example
//!
//! ```
//! use oximedia_forensics::custody::ChainOfCustody;
//!
//! let mut chain = ChainOfCustody::new(42);
//! chain.add_event("ingest", 1001, 1_700_000_000);
//! chain.add_event("transcode", 1002, 1_700_000_100);
//! chain.add_event("deliver", 1003, 1_700_000_200);
//! assert!(chain.verify());
//! ```

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// FNV-1a 64-bit hasher (no external dependency)
// ---------------------------------------------------------------------------

const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
const FNV_PRIME: u64 = 1_099_511_628_211;

fn fnv1a_64(data: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn hash_event(
    prev_checksum: u64,
    asset_id: u64,
    seq: u64,
    actor: u64,
    action: &str,
    ts: u64,
) -> u64 {
    let mut buf = Vec::with_capacity(64);
    buf.extend_from_slice(&prev_checksum.to_le_bytes());
    buf.extend_from_slice(&asset_id.to_le_bytes());
    buf.extend_from_slice(&seq.to_le_bytes());
    buf.extend_from_slice(&actor.to_le_bytes());
    buf.extend_from_slice(action.as_bytes());
    buf.extend_from_slice(&ts.to_le_bytes());
    fnv1a_64(&buf)
}

// ---------------------------------------------------------------------------
// CustodyEvent
// ---------------------------------------------------------------------------

/// A single immutable entry in the chain of custody.
#[derive(Debug, Clone)]
pub struct CustodyEvent {
    /// Sequence number within the chain (0-based).
    pub sequence: u64,
    /// Identifier of the actor who performed the action.
    pub actor: u64,
    /// Human-readable action description (e.g. `"ingest"`, `"transcode"`).
    pub action: String,
    /// Unix timestamp (seconds) when the action occurred.
    pub timestamp: u64,
    /// FNV-1a checksum of this event (computed at insertion time).
    pub checksum: u64,
    /// Checksum of the immediately preceding event (`0` for the first event).
    pub prev_checksum: u64,
}

// ---------------------------------------------------------------------------
// ChainOfCustody
// ---------------------------------------------------------------------------

/// A tamper-evident chain of custody for a media asset.
#[derive(Debug, Clone)]
pub struct ChainOfCustody {
    /// The asset this chain belongs to.
    pub asset_id: u64,
    /// Ordered list of custody events.
    pub events: Vec<CustodyEvent>,
}

impl ChainOfCustody {
    /// Create a new, empty chain for `asset_id`.
    #[must_use]
    pub fn new(asset_id: u64) -> Self {
        Self {
            asset_id,
            events: Vec::new(),
        }
    }

    /// Append a new event to the chain.
    ///
    /// The event's checksum is derived from the previous event's checksum
    /// (or `0` if this is the first event) and the event payload.
    pub fn add_event(&mut self, action: &str, actor: u64, ts: u64) {
        let seq = self.events.len() as u64;
        let prev_checksum = self
            .events
            .last()
            .map(|e| e.checksum)
            .unwrap_or(0);

        let checksum = hash_event(prev_checksum, self.asset_id, seq, actor, action, ts);

        self.events.push(CustodyEvent {
            sequence: seq,
            actor,
            action: action.to_string(),
            timestamp: ts,
            checksum,
            prev_checksum,
        });
    }

    /// Verify the integrity of the entire chain.
    ///
    /// Checks:
    /// - The genesis event has `prev_checksum == 0`.
    /// - Each subsequent event's `prev_checksum` matches the computed checksum
    ///   of the preceding event.
    /// - Timestamps are non-decreasing.
    ///
    /// Returns `true` when the chain is intact.  An empty chain is considered
    /// trivially valid (`true`).
    #[must_use]
    pub fn verify(&self) -> bool {
        if self.events.is_empty() {
            return true;
        }

        // Recompute checksum for each event and validate linkage
        for (i, event) in self.events.iter().enumerate() {
            // Timestamp monotonicity
            if i > 0 && event.timestamp < self.events[i - 1].timestamp {
                return false;
            }

            // Expected previous checksum
            let expected_prev = if i == 0 {
                0u64
            } else {
                self.events[i - 1].checksum
            };

            if event.prev_checksum != expected_prev {
                return false;
            }

            // Recompute this event's checksum
            let recomputed = hash_event(
                event.prev_checksum,
                self.asset_id,
                event.sequence,
                event.actor,
                &event.action,
                event.timestamp,
            );

            if recomputed != event.checksum {
                return false;
            }
        }

        true
    }

    /// Number of events in the chain.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns `true` when the chain has no events.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Return the most recent event, if any.
    #[must_use]
    pub fn latest(&self) -> Option<&CustodyEvent> {
        self.events.last()
    }

    /// Find events by action name.
    #[must_use]
    pub fn find_by_action(&self, action: &str) -> Vec<&CustodyEvent> {
        self.events.iter().filter(|e| e.action == action).collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn build_chain(events: &[(&str, u64, u64)]) -> ChainOfCustody {
        let mut chain = ChainOfCustody::new(99);
        for &(action, actor, ts) in events {
            chain.add_event(action, actor, ts);
        }
        chain
    }

    // ── new ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_new_starts_empty() {
        let chain = ChainOfCustody::new(1);
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
    }

    // ── add_event ────────────────────────────────────────────────────────────

    #[test]
    fn test_add_event_increments_length() {
        let mut chain = ChainOfCustody::new(1);
        chain.add_event("ingest", 100, 1000);
        assert_eq!(chain.len(), 1);
        chain.add_event("transcode", 101, 2000);
        assert_eq!(chain.len(), 2);
    }

    #[test]
    fn test_first_event_prev_checksum_is_zero() {
        let mut chain = ChainOfCustody::new(1);
        chain.add_event("start", 1, 1000);
        assert_eq!(chain.events[0].prev_checksum, 0);
    }

    #[test]
    fn test_second_event_links_to_first() {
        let mut chain = ChainOfCustody::new(1);
        chain.add_event("a", 1, 1000);
        chain.add_event("b", 2, 2000);
        assert_eq!(chain.events[1].prev_checksum, chain.events[0].checksum);
    }

    #[test]
    fn test_sequence_numbers_are_contiguous() {
        let chain = build_chain(&[("a", 1, 1), ("b", 2, 2), ("c", 3, 3)]);
        for (i, e) in chain.events.iter().enumerate() {
            assert_eq!(e.sequence, i as u64);
        }
    }

    // ── verify ───────────────────────────────────────────────────────────────

    #[test]
    fn test_verify_empty_chain() {
        let chain = ChainOfCustody::new(1);
        assert!(chain.verify());
    }

    #[test]
    fn test_verify_single_event() {
        let mut chain = ChainOfCustody::new(1);
        chain.add_event("ingest", 1, 1000);
        assert!(chain.verify());
    }

    #[test]
    fn test_verify_multi_event_chain() {
        let chain = build_chain(&[
            ("ingest", 1001, 1_700_000_000),
            ("transcode", 1002, 1_700_000_100),
            ("deliver", 1003, 1_700_000_200),
        ]);
        assert!(chain.verify());
    }

    #[test]
    fn test_verify_fails_on_tampered_checksum() {
        let mut chain = build_chain(&[("ingest", 1, 1000), ("deliver", 2, 2000)]);
        // Corrupt the first event's checksum
        chain.events[0].checksum ^= 0xDEAD_BEEF;
        assert!(!chain.verify());
    }

    #[test]
    fn test_verify_fails_on_tampered_prev_checksum() {
        let mut chain = build_chain(&[("a", 1, 1000), ("b", 2, 2000)]);
        chain.events[1].prev_checksum ^= 1;
        assert!(!chain.verify());
    }

    #[test]
    fn test_verify_fails_on_tampered_action() {
        let mut chain = build_chain(&[("ingest", 1, 1000)]);
        chain.events[0].action = "TAMPERED".to_string();
        assert!(!chain.verify());
    }

    #[test]
    fn test_verify_fails_on_out_of_order_timestamps() {
        let mut chain = ChainOfCustody::new(1);
        // Manually construct events with decreasing timestamps to bypass add_event ordering
        let e1 = CustodyEvent {
            sequence: 0,
            actor: 1,
            action: "first".to_string(),
            timestamp: 2000,
            checksum: hash_event(0, 1, 0, 1, "first", 2000),
            prev_checksum: 0,
        };
        let prev = e1.checksum;
        let e2 = CustodyEvent {
            sequence: 1,
            actor: 2,
            action: "second".to_string(),
            timestamp: 1000, // earlier than first — invalid
            checksum: hash_event(prev, 1, 1, 2, "second", 1000),
            prev_checksum: prev,
        };
        chain.events.push(e1);
        chain.events.push(e2);
        assert!(!chain.verify());
    }

    // ── utility methods ───────────────────────────────────────────────────────

    #[test]
    fn test_latest_returns_last_event() {
        let chain = build_chain(&[("a", 1, 1), ("b", 2, 2)]);
        let latest = chain.latest().expect("should have a latest event");
        assert_eq!(latest.action, "b");
    }

    #[test]
    fn test_latest_empty_returns_none() {
        let chain = ChainOfCustody::new(1);
        assert!(chain.latest().is_none());
    }

    #[test]
    fn test_find_by_action() {
        let chain = build_chain(&[("ingest", 1, 1), ("transcode", 2, 2), ("ingest", 3, 3)]);
        let found = chain.find_by_action("ingest");
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn test_find_by_action_not_found() {
        let chain = build_chain(&[("ingest", 1, 1)]);
        let found = chain.find_by_action("nonexistent");
        assert!(found.is_empty());
    }
}
