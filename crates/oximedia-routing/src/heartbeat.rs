//! NMOS node heartbeat tracking.
//!
//! NMOS IS-04 requires registered nodes to send periodic heartbeat messages
//! to the registry.  If the registry does not receive a heartbeat within the
//! configured timeout window it removes the node's resources from the
//! registry.
//!
//! [`NodeHeartbeat`] provides a lightweight, clock-agnostic implementation:
//! the caller supplies monotonic timestamps (e.g., seconds since Unix epoch)
//! and [`is_alive`](NodeHeartbeat::is_alive) checks whether the node is still considered healthy.
//!
//! # Example
//!
//! ```
//! use oximedia_routing::heartbeat::NodeHeartbeat;
//!
//! let mut hb = NodeHeartbeat::new("node-001");
//! hb.update(1000);   // heartbeat received at t=1000
//!
//! assert!(hb.is_alive(1005, 12));  // 5 s elapsed, timeout = 12 → alive
//! assert!(!hb.is_alive(1013, 12)); // 13 s elapsed → dead
//! ```

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Tracks the last heartbeat timestamp for a single NMOS node.
#[derive(Debug, Clone)]
pub struct NodeHeartbeat {
    /// The identifier of the node being tracked.
    pub node_id: String,
    /// The timestamp (caller-defined units, typically seconds) of the last
    /// received heartbeat.  `None` means no heartbeat has been received yet.
    last_seen: Option<u64>,
}

impl NodeHeartbeat {
    /// Create a new heartbeat tracker for `node_id`.
    ///
    /// The initial state is "never seen" — [`is_alive`] will return `false`
    /// until the first [`update`] call.
    ///
    /// [`is_alive`]: NodeHeartbeat::is_alive
    /// [`update`]: NodeHeartbeat::update
    pub fn new(node_id: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            last_seen: None,
        }
    }

    /// Record a heartbeat at timestamp `ts`.
    ///
    /// If `ts` is earlier than the previously recorded timestamp the update
    /// is **ignored** (timestamps must be monotonically non-decreasing).
    pub fn update(&mut self, ts: u64) {
        match self.last_seen {
            None => self.last_seen = Some(ts),
            Some(prev) if ts >= prev => self.last_seen = Some(ts),
            _ => {} // out-of-order / older timestamp — ignore
        }
    }

    /// Return the timestamp of the last received heartbeat, if any.
    pub fn last_seen(&self) -> Option<u64> {
        self.last_seen
    }

    /// Check whether the node is still considered alive at `now`.
    ///
    /// # Parameters
    ///
    /// * `now`     — current timestamp in the same units used for [`update`].
    /// * `timeout` — maximum acceptable gap between `last_seen` and `now`.
    ///
    /// # Returns
    ///
    /// `true`  when a heartbeat was received **and** `now - last_seen ≤ timeout`.
    /// `false` when no heartbeat has ever been received, or the last heartbeat
    ///         is older than `timeout` units.
    ///
    /// [`update`]: NodeHeartbeat::update
    pub fn is_alive(&self, now: u64, timeout: u64) -> bool {
        match self.last_seen {
            None => false,
            Some(last) => now.saturating_sub(last) <= timeout,
        }
    }

    /// Reset the tracker to the "never seen" state.
    pub fn reset(&mut self) {
        self.last_seen = None;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_is_not_alive() {
        let hb = NodeHeartbeat::new("node-1");
        assert!(!hb.is_alive(1000, 12), "brand-new node should not be alive");
        assert_eq!(hb.last_seen(), None);
    }

    #[test]
    fn test_update_then_alive() {
        let mut hb = NodeHeartbeat::new("node-1");
        hb.update(100);
        assert!(hb.is_alive(100, 12), "alive immediately after update");
        assert!(hb.is_alive(112, 12), "alive exactly at timeout boundary");
    }

    #[test]
    fn test_expired_heartbeat() {
        let mut hb = NodeHeartbeat::new("node-2");
        hb.update(100);
        assert!(
            !hb.is_alive(113, 12),
            "13 s elapsed with 12 s timeout → dead"
        );
    }

    #[test]
    fn test_update_advances_timestamp() {
        let mut hb = NodeHeartbeat::new("n");
        hb.update(50);
        hb.update(100);
        assert_eq!(hb.last_seen(), Some(100));
        // Now alive at t=110 with timeout=12 (only 10 s since t=100)
        assert!(hb.is_alive(110, 12));
    }

    #[test]
    fn test_out_of_order_update_ignored() {
        let mut hb = NodeHeartbeat::new("n");
        hb.update(200);
        hb.update(100); // older timestamp — should be ignored
        assert_eq!(hb.last_seen(), Some(200));
    }

    #[test]
    fn test_update_same_timestamp_accepted() {
        let mut hb = NodeHeartbeat::new("n");
        hb.update(100);
        hb.update(100);
        assert_eq!(hb.last_seen(), Some(100));
    }

    #[test]
    fn test_reset_clears_state() {
        let mut hb = NodeHeartbeat::new("n");
        hb.update(500);
        hb.reset();
        assert_eq!(hb.last_seen(), None);
        assert!(!hb.is_alive(500, 30));
    }

    #[test]
    fn test_zero_timeout_alive_only_at_exact_timestamp() {
        let mut hb = NodeHeartbeat::new("n");
        hb.update(42);
        assert!(
            hb.is_alive(42, 0),
            "alive exactly at timestamp with timeout=0"
        );
        assert!(!hb.is_alive(43, 0), "1 unit after with timeout=0 → dead");
    }

    #[test]
    fn test_node_id_stored() {
        let hb = NodeHeartbeat::new("unique-node-xyz");
        assert_eq!(hb.node_id, "unique-node-xyz");
    }

    #[test]
    fn test_saturating_sub_no_overflow() {
        let mut hb = NodeHeartbeat::new("n");
        hb.update(u64::MAX);
        // now < last_seen after saturating_sub → difference = 0 → alive
        assert!(hb.is_alive(0, 100));
    }
}
