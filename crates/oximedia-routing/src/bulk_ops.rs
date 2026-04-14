//! Bulk activation of NMOS IS-05 connections.
//!
//! IS-05 defines a bulk-activation endpoint that allows a controller to
//! activate multiple sender→receiver connections atomically.  This module
//! provides [`BulkActivator`] which accumulates `(sender_id, receiver_id)`
//! pairs and then attempts to activate them all, returning per-connection
//! results.
//!
//! # Example
//!
//! ```
//! use oximedia_routing::bulk_ops::BulkActivator;
//!
//! let mut activator = BulkActivator::new();
//! activator.add("sender-1", "receiver-1");
//! activator.add("sender-2", "receiver-2");
//!
//! let results = activator.activate_all();
//! assert_eq!(results.len(), 2);
//! for r in &results {
//!     assert!(r.is_ok(), "activation should succeed: {:?}", r);
//! }
//! ```

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A pending connection between a sender and a receiver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingConnection {
    /// NMOS sender resource identifier.
    pub sender_id: String,
    /// NMOS receiver resource identifier.
    pub receiver_id: String,
}

impl PendingConnection {
    /// Create a new pending connection.
    pub fn new(sender_id: impl Into<String>, receiver_id: impl Into<String>) -> Self {
        Self {
            sender_id: sender_id.into(),
            receiver_id: receiver_id.into(),
        }
    }
}

/// Accumulates sender→receiver connection requests and activates them in bulk.
///
/// After calling [`activate_all`][BulkActivator::activate_all] the queue is
/// **consumed** — subsequent calls will return an empty result unless new
/// connections are added with [`add`][BulkActivator::add].
#[derive(Debug, Default)]
pub struct BulkActivator {
    pending: Vec<PendingConnection>,
}

impl BulkActivator {
    /// Create an empty bulk activator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enqueue a connection from `sender_id` to `receiver_id`.
    pub fn add(&mut self, sender_id: impl Into<String>, receiver_id: impl Into<String>) {
        self.pending
            .push(PendingConnection::new(sender_id, receiver_id));
    }

    /// Attempt to activate all queued connections and return per-connection results.
    ///
    /// # Activation logic
    ///
    /// A connection is considered successful when both IDs are non-empty strings.
    /// In a real IS-05 implementation this would contact the NMOS registry; here
    /// we validate the IDs locally so the module can be tested without network
    /// access.
    ///
    /// # Returns
    ///
    /// A `Vec<Result<(), String>>` with one entry per queued connection, in
    /// insertion order.  After this call the internal queue is cleared.
    pub fn activate_all(&mut self) -> Vec<Result<(), String>> {
        let connections = std::mem::take(&mut self.pending);
        connections.into_iter().map(activate_connection).collect()
    }

    /// Return the number of pending (not yet activated) connections.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Discard all pending connections without activating them.
    pub fn clear(&mut self) {
        self.pending.clear();
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Validate and activate a single connection.
///
/// Returns `Err` with a descriptive message when either ID is empty, or when
/// the sender and receiver IDs are identical (a loop-back, which IS-05
/// implementations should reject).
fn activate_connection(conn: PendingConnection) -> Result<(), String> {
    if conn.sender_id.is_empty() {
        return Err("sender_id must not be empty".to_owned());
    }
    if conn.receiver_id.is_empty() {
        return Err("receiver_id must not be empty".to_owned());
    }
    if conn.sender_id == conn.receiver_id {
        return Err(format!(
            "sender_id and receiver_id are identical: {:?}",
            conn.sender_id
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_has_no_pending() {
        let a = BulkActivator::new();
        assert_eq!(a.pending_count(), 0);
    }

    #[test]
    fn test_add_increments_pending() {
        let mut a = BulkActivator::new();
        a.add("s1", "r1");
        a.add("s2", "r2");
        assert_eq!(a.pending_count(), 2);
    }

    #[test]
    fn test_activate_all_returns_one_result_per_connection() {
        let mut a = BulkActivator::new();
        a.add("sender-1", "receiver-1");
        a.add("sender-2", "receiver-2");
        let results = a.activate_all();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_activate_all_clears_queue() {
        let mut a = BulkActivator::new();
        a.add("s", "r");
        let _ = a.activate_all();
        assert_eq!(a.pending_count(), 0);
        let results = a.activate_all();
        assert!(results.is_empty(), "queue should be empty after activation");
    }

    #[test]
    fn test_valid_connections_succeed() {
        let mut a = BulkActivator::new();
        a.add("sender-001", "receiver-001");
        a.add("source-A", "sink-B");
        for r in a.activate_all() {
            assert!(r.is_ok(), "expected Ok, got: {r:?}");
        }
    }

    #[test]
    fn test_empty_sender_returns_err() {
        let mut a = BulkActivator::new();
        a.add("", "receiver-1");
        let results = a.activate_all();
        assert_eq!(results.len(), 1);
        assert!(results[0].is_err(), "empty sender should fail");
    }

    #[test]
    fn test_empty_receiver_returns_err() {
        let mut a = BulkActivator::new();
        a.add("sender-1", "");
        let results = a.activate_all();
        assert!(results[0].is_err(), "empty receiver should fail");
    }

    #[test]
    fn test_identical_ids_returns_err() {
        let mut a = BulkActivator::new();
        a.add("same-id", "same-id");
        let results = a.activate_all();
        assert!(results[0].is_err(), "loop-back connection should fail");
    }

    #[test]
    fn test_mixed_results() {
        let mut a = BulkActivator::new();
        a.add("good-sender", "good-receiver"); // ok
        a.add("", "receiver"); // err: empty sender
        a.add("s", "r"); // ok
        let results = a.activate_all();
        assert_eq!(results.len(), 3);
        assert!(results[0].is_ok());
        assert!(results[1].is_err());
        assert!(results[2].is_ok());
    }

    #[test]
    fn test_clear_removes_pending() {
        let mut a = BulkActivator::new();
        a.add("s", "r");
        a.clear();
        assert_eq!(a.pending_count(), 0);
    }

    #[test]
    fn test_pending_connection_equality() {
        let c1 = PendingConnection::new("s", "r");
        let c2 = PendingConnection::new("s", "r");
        assert_eq!(c1, c2);
    }
}
