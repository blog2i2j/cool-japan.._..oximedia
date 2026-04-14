//! Enhanced distributed primitives: Raft vote, work-stealing queue,
//! backpressure controller, distributed checkpointing, consistent hash ring,
//! distributed circuit breaker, shard allocator, service registry, and
//! replication manager.

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Raft primitives
// ---------------------------------------------------------------------------

/// Persistent and volatile state for a Raft node (u64-keyed variant).
#[derive(Debug)]
pub struct RaftState {
    /// Latest term this node has seen.
    pub current_term: u64,
    /// Node ID of the candidate this node voted for in the current term.
    pub voted_for: Option<u64>,
    /// Replicated log entries.
    pub log: Vec<LogEntry>,
}

/// A single entry in the Raft log.
#[derive(Debug, Clone)]
pub struct LogEntry {
    /// Term in which the entry was created.
    pub term: u64,
    /// 1-based log index.
    pub index: u64,
    /// Encoded command payload.
    pub command: String,
}

impl RaftState {
    /// Create a new Raft state at term 0.
    #[must_use]
    pub fn new() -> Self {
        Self {
            current_term: 0,
            voted_for: None,
            log: Vec::new(),
        }
    }
}

impl Default for RaftState {
    fn default() -> Self {
        Self::new()
    }
}

/// Response to a `RequestVote` RPC.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoteResponse {
    /// The term at which the vote was evaluated.
    pub term: u64,
    /// Whether the vote was granted.
    pub vote_granted: bool,
}

/// A single Raft node.
#[derive(Debug)]
pub struct RaftNode {
    /// This node's unique ID.
    pub node_id: u64,
    /// Raft state (wrapped for interior mutability in concurrent use).
    state: Mutex<RaftState>,
}

impl RaftNode {
    /// Create a new Raft node with the given ID.
    #[must_use]
    pub fn new(node_id: u64) -> Self {
        Self {
            node_id,
            state: Mutex::new(RaftState::new()),
        }
    }

    /// Handle a `RequestVote` RPC from a candidate.
    ///
    /// Implements the Raft voting rules:
    /// - If `term < current_term`, deny the vote.
    /// - If `term > current_term`, update the term and clear any prior vote.
    /// - Grant the vote if `voted_for` is `None` or already equals `candidate_id`,
    ///   **and** the candidate's log is at least as up-to-date as ours
    ///   (last log index and term comparison).
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned (should never happen in normal use).
    pub fn request_vote(
        &self,
        term: u64,
        candidate_id: u64,
        last_log_index: u64,
        last_log_term: u64,
    ) -> VoteResponse {
        let mut state = self.state.lock().expect("raft state mutex poisoned");

        // If we see a higher term, update and clear our vote.
        if term > state.current_term {
            state.current_term = term;
            state.voted_for = None;
        }

        // Deny if the candidate's term is stale.
        if term < state.current_term {
            return VoteResponse {
                term: state.current_term,
                vote_granted: false,
            };
        }

        // Check whether we have already voted for someone else this term.
        let can_vote = state.voted_for.is_none() || state.voted_for == Some(candidate_id);
        if !can_vote {
            return VoteResponse {
                term: state.current_term,
                vote_granted: false,
            };
        }

        // Check log up-to-date-ness (§5.4.1 of the Raft paper).
        let our_last_term = state.log.last().map_or(0, |e| e.term);
        let our_last_index = state.log.len() as u64;

        let candidate_log_ok = if last_log_term != our_last_term {
            last_log_term > our_last_term
        } else {
            last_log_index >= our_last_index
        };

        if candidate_log_ok {
            state.voted_for = Some(candidate_id);
            VoteResponse {
                term: state.current_term,
                vote_granted: true,
            }
        } else {
            VoteResponse {
                term: state.current_term,
                vote_granted: false,
            }
        }
    }

    /// Return the current term.
    pub fn current_term(&self) -> u64 {
        self.state
            .lock()
            .expect("raft state mutex poisoned")
            .current_term
    }

    /// Return who we voted for in the current term, if anyone.
    pub fn voted_for(&self) -> Option<u64> {
        self.state
            .lock()
            .expect("raft state mutex poisoned")
            .voted_for
    }
}

// ---------------------------------------------------------------------------
// Work-stealing queue
// ---------------------------------------------------------------------------

/// A per-worker work-stealing deque.
///
/// The owner pushes/pops from the back; thieves steal from the front.
#[derive(Debug)]
pub struct WorkStealingQueue<T> {
    /// Tasks owned by this worker (LIFO end = back).
    local: Vec<T>,
    /// Tasks stolen from other workers (to be processed next).
    stolen: Vec<T>,
}

impl<T> WorkStealingQueue<T> {
    /// Create an empty queue.
    #[must_use]
    pub fn new() -> Self {
        Self {
            local: Vec::new(),
            stolen: Vec::new(),
        }
    }

    /// Push a task onto the local (owner's) end.
    pub fn push(&mut self, item: T) {
        self.local.push(item);
    }

    /// Pop a task for the owner to execute.
    ///
    /// Checks the `stolen` buffer first (so stolen tasks are prioritised),
    /// then falls back to the local deque (LIFO).
    pub fn pop(&mut self) -> Option<T> {
        if let Some(item) = self.stolen.pop() {
            return Some(item);
        }
        self.local.pop()
    }

    /// Steal a task from the front of the local deque (FIFO).
    ///
    /// Returns `None` if the local deque is empty.
    pub fn steal(&mut self) -> Option<T> {
        if self.local.is_empty() {
            None
        } else {
            Some(self.local.remove(0))
        }
    }

    /// Number of tasks in the local deque (not counting stolen tasks).
    #[must_use]
    pub fn len(&self) -> usize {
        self.local.len() + self.stolen.len()
    }

    /// Returns `true` if both the local and stolen buffers are empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.local.is_empty() && self.stolen.is_empty()
    }
}

impl<T> Default for WorkStealingQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// BackpressureController
// ---------------------------------------------------------------------------

/// A simple pending-count-based backpressure controller.
///
/// `try_submit` returns `true` only when the number of in-flight items is
/// strictly below `max_pending`. `complete_one` decrements the counter.
#[derive(Debug)]
pub struct BackpressureController {
    /// Maximum number of simultaneously in-flight items.
    max_pending: usize,
    /// Current count of in-flight items.
    pending: AtomicUsize,
}

impl BackpressureController {
    /// Create a controller with the given maximum pending count.
    #[must_use]
    pub fn new(max_pending: usize) -> Self {
        Self {
            max_pending,
            pending: AtomicUsize::new(0),
        }
    }

    /// Attempt to submit a new item.
    ///
    /// Returns `true` and increments the pending counter if the limit has not
    /// been reached. Returns `false` without modifying state if the queue is full.
    pub fn try_submit(&self) -> bool {
        // Use a compare-and-swap loop to atomically increment only if below limit.
        loop {
            let current = self.pending.load(Ordering::Acquire);
            if current >= self.max_pending {
                return false;
            }
            match self.pending.compare_exchange(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(_) => continue, // raced, retry
            }
        }
    }

    /// Decrement the pending counter when an item completes.
    ///
    /// Will not decrement below zero (saturating).
    pub fn complete_one(&self) {
        let _ = self
            .pending
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| {
                if v > 0 {
                    Some(v - 1)
                } else {
                    None
                }
            });
    }

    /// Current number of in-flight items.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.load(Ordering::Acquire)
    }

    /// Maximum allowed pending count.
    #[must_use]
    pub fn max_pending(&self) -> usize {
        self.max_pending
    }
}

// ---------------------------------------------------------------------------
// Distributed checkpointing
// ---------------------------------------------------------------------------

/// A single distributed checkpoint snapshot.
#[derive(Debug, Clone)]
pub struct DistributedCheckpoint {
    /// The node that created this checkpoint.
    pub node_id: u64,
    /// Monotonically increasing sequence number (unique per node).
    pub sequence: u64,
    /// Opaque serialised state data.
    pub state: Vec<u8>,
}

/// Coordinates checkpoint creation and storage across multiple nodes.
#[derive(Debug, Default)]
pub struct CheckpointCoordinator {
    /// Per-node sequence counters.
    sequences: HashMap<u64, u64>,
    /// All stored checkpoints.
    checkpoints: Vec<DistributedCheckpoint>,
}

impl CheckpointCoordinator {
    /// Create a new coordinator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Take a checkpoint for `node_id` with the given state.
    ///
    /// Allocates the next sequence number for the node and stores the
    /// checkpoint. Returns the assigned sequence number.
    pub fn take_checkpoint(&mut self, node_id: u64, state: &[u8]) -> u64 {
        let seq = self.sequences.entry(node_id).or_insert(0);
        *seq += 1;
        let sequence = *seq;

        self.checkpoints.push(DistributedCheckpoint {
            node_id,
            sequence,
            state: state.to_vec(),
        });

        sequence
    }

    /// Retrieve the most recent checkpoint for a node.
    #[must_use]
    pub fn latest_checkpoint(&self, node_id: u64) -> Option<&DistributedCheckpoint> {
        self.checkpoints
            .iter()
            .filter(|c| c.node_id == node_id)
            .max_by_key(|c| c.sequence)
    }

    /// Total number of stored checkpoints.
    #[must_use]
    pub fn checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }
}

// ---------------------------------------------------------------------------
// Consistent hash ring
// ---------------------------------------------------------------------------

/// A consistent hash ring with virtual-node support for even key distribution.
///
/// Uses the FNV-1a hash algorithm for deterministic, dependency-free hashing.
#[derive(Debug)]
pub struct ConsistentHashRing {
    /// Number of virtual nodes per physical node.
    virtual_nodes: u32,
    /// Sorted ring: (hash_position, node_id).
    ring: Vec<(u64, u64)>,
    /// Set of registered physical nodes.
    nodes: Vec<u64>,
}

impl ConsistentHashRing {
    /// Create a new ring with the given number of virtual nodes per physical node.
    #[must_use]
    pub fn new(virtual_nodes: u32) -> Self {
        Self {
            virtual_nodes,
            ring: Vec::new(),
            nodes: Vec::new(),
        }
    }

    /// Add a physical node to the ring.
    pub fn add_node(&mut self, id: u64) {
        if self.nodes.contains(&id) {
            return;
        }
        self.nodes.push(id);
        for i in 0..self.virtual_nodes {
            let key = format!("{id}:vn:{i}");
            let h = Self::fnv1a(key.as_bytes());
            self.ring.push((h, id));
        }
        self.ring.sort_unstable_by_key(|(h, _)| *h);
    }

    /// Remove a physical node from the ring.
    pub fn remove_node(&mut self, id: u64) {
        self.nodes.retain(|&n| n != id);
        for i in 0..self.virtual_nodes {
            let key = format!("{id}:vn:{i}");
            let h = Self::fnv1a(key.as_bytes());
            self.ring.retain(|(rh, _)| *rh != h);
        }
    }

    /// Look up which physical node owns the given key.
    ///
    /// Returns `None` if the ring is empty.
    #[must_use]
    pub fn get_node(&self, key: &[u8]) -> Option<u64> {
        if self.ring.is_empty() {
            return None;
        }
        let h = Self::fnv1a(key);
        // Binary-search for the first ring entry >= h.
        match self.ring.binary_search_by_key(&h, |(rh, _)| *rh) {
            Ok(idx) => Some(self.ring[idx].1),
            Err(idx) => {
                // Wrap around to first entry if h > all ring entries.
                let idx = idx % self.ring.len();
                Some(self.ring[idx].1)
            }
        }
    }

    /// Number of registered physical nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// FNV-1a 64-bit hash.
    fn fnv1a(data: &[u8]) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for &byte in data {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
        hash
    }
}

// ---------------------------------------------------------------------------
// Distributed circuit breaker
// ---------------------------------------------------------------------------

/// A distributed circuit breaker with failure threshold and timeout.
#[derive(Debug)]
pub struct DistributedCircuitBreaker {
    /// Number of consecutive failures that trip the circuit.
    threshold: u32,
    /// How long the circuit stays open (milliseconds).
    timeout_ms: u64,
    /// Consecutive failure count.
    failures: AtomicU64,
    /// Timestamp (ms) when the circuit was opened (0 = not open).
    opened_at_ms: AtomicU64,
    /// Whether the circuit is currently open.
    open: AtomicBool,
}

impl DistributedCircuitBreaker {
    /// Create a new circuit breaker.
    #[must_use]
    pub fn new(threshold: u32, timeout_ms: u64) -> Self {
        Self {
            threshold,
            timeout_ms,
            failures: AtomicU64::new(0),
            opened_at_ms: AtomicU64::new(0),
            open: AtomicBool::new(false),
        }
    }

    /// Record a successful call. Resets the failure count if the circuit is
    /// closed.
    pub fn call_succeeded(&self) {
        if !self.open.load(Ordering::Acquire) {
            self.failures.store(0, Ordering::Release);
        }
    }

    /// Record a failed call. Opens the circuit if the failure threshold is
    /// reached.
    pub fn call_failed(&self) {
        let prev = self.failures.fetch_add(1, Ordering::AcqRel);
        if prev + 1 >= u64::from(self.threshold) {
            let now_ms = Self::now_ms();
            self.opened_at_ms.store(now_ms, Ordering::Release);
            self.open.store(true, Ordering::Release);
        }
    }

    /// Returns `true` if the circuit is currently open (requests should be
    /// rejected).
    ///
    /// If the circuit was opened more than `timeout_ms` ago, it automatically
    /// transitions back to closed (allowing probe requests through).
    pub fn is_open(&self) -> bool {
        if !self.open.load(Ordering::Acquire) {
            return false;
        }
        // Check whether the timeout has elapsed.
        let opened_at = self.opened_at_ms.load(Ordering::Acquire);
        let elapsed = Self::now_ms().saturating_sub(opened_at);
        if elapsed >= self.timeout_ms {
            // Transition to closed (half-open probe).
            self.open.store(false, Ordering::Release);
            self.failures.store(0, Ordering::Release);
            return false;
        }
        true
    }

    /// Current consecutive failure count.
    #[must_use]
    pub fn failure_count(&self) -> u64 {
        self.failures.load(Ordering::Acquire)
    }

    /// Reset the circuit breaker to closed state.
    pub fn reset(&self) {
        self.failures.store(0, Ordering::Release);
        self.open.store(false, Ordering::Release);
        self.opened_at_ms.store(0, Ordering::Release);
    }

    /// Returns the current time as milliseconds since the Unix epoch.
    /// Falls back to 0 on error (should never happen on a healthy system).
    fn now_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Shard allocation
// ---------------------------------------------------------------------------

/// Assign a key hash to a shard using simple modulo sharding.
#[must_use]
pub fn shard_assign(key_hash: u64, num_shards: u32) -> u32 {
    if num_shards == 0 {
        return 0;
    }
    (key_hash % u64::from(num_shards)) as u32
}

/// A simple FNV-1a hash helper for byte slices.
#[must_use]
pub fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

/// A fixed-shard key-value map.
///
/// Keys are hashed with FNV-1a, then assigned to a shard via modulo. Each
/// shard holds an independent `HashMap` to allow future parallelism.
#[derive(Debug)]
pub struct SimpleShardMap {
    shards: Vec<HashMap<Vec<u8>, Vec<u8>>>,
    num_shards: u32,
}

impl SimpleShardMap {
    /// Create a new shard map with `num_shards` shards.
    #[must_use]
    pub fn new(num_shards: u32) -> Self {
        let count = num_shards.max(1) as usize;
        Self {
            shards: vec![HashMap::new(); count],
            num_shards: num_shards.max(1),
        }
    }

    /// Insert a key-value pair.
    pub fn insert(&mut self, key: &[u8], value: Vec<u8>) {
        let h = fnv1a_hash(key);
        let shard = shard_assign(h, self.num_shards) as usize;
        self.shards[shard].insert(key.to_vec(), value);
    }

    /// Get a value by key.
    #[must_use]
    pub fn get(&self, key: &[u8]) -> Option<&[u8]> {
        let h = fnv1a_hash(key);
        let shard = shard_assign(h, self.num_shards) as usize;
        self.shards[shard].get(key).map(|v| v.as_slice())
    }

    /// Number of entries across all shards.
    #[must_use]
    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.len()).sum()
    }

    /// Returns `true` if the map has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns entries-per-shard counts for load analysis.
    #[must_use]
    pub fn shard_counts(&self) -> Vec<usize> {
        self.shards.iter().map(|s| s.len()).collect()
    }
}

// ---------------------------------------------------------------------------
// Service registry (TTL-based)
// ---------------------------------------------------------------------------

/// A registered service endpoint with TTL tracking.
#[derive(Debug, Clone)]
struct ServiceEntry {
    /// Network address string (e.g. "192.168.1.10:50052").
    addr: String,
    /// Monotonic instant when this registration expires.
    expires_at: Instant,
}

/// A simple in-process service registry with TTL-based expiry.
#[derive(Debug)]
pub struct ServiceRegistry {
    entries: Mutex<HashMap<u64, ServiceEntry>>,
    /// Default TTL for new registrations.
    default_ttl: Duration,
}

impl ServiceRegistry {
    /// Create a registry with the given default TTL.
    #[must_use]
    pub fn new(default_ttl: Duration) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            default_ttl,
        }
    }

    /// Create a registry with a 60-second default TTL.
    #[must_use]
    pub fn with_default_ttl() -> Self {
        Self::new(Duration::from_secs(60))
    }

    /// Register a service endpoint.
    ///
    /// If a registration for `service_id` already exists it is replaced.
    pub fn register(&self, service_id: u64, addr: &str) {
        let entry = ServiceEntry {
            addr: addr.to_string(),
            expires_at: Instant::now() + self.default_ttl,
        };
        self.entries
            .lock()
            .expect("service registry lock poisoned")
            .insert(service_id, entry);
    }

    /// Discover the address of a registered service.
    ///
    /// Returns `None` if the service is not registered or has expired.
    pub fn discover(&self, service_id: u64) -> Option<String> {
        let mut guard = self.entries.lock().expect("service registry lock poisoned");
        match guard.get(&service_id) {
            Some(entry) if entry.expires_at > Instant::now() => Some(entry.addr.clone()),
            Some(_) => {
                // Expired — remove and return None.
                guard.remove(&service_id);
                None
            }
            None => None,
        }
    }

    /// Number of (potentially expired) registered services.
    #[must_use]
    pub fn registered_count(&self) -> usize {
        self.entries
            .lock()
            .expect("service registry lock poisoned")
            .len()
    }

    /// Purge all expired entries.
    pub fn evict_expired(&self) {
        let now = Instant::now();
        self.entries
            .lock()
            .expect("service registry lock poisoned")
            .retain(|_, e| e.expires_at > now);
    }
}

// ---------------------------------------------------------------------------
// Replication manager
// ---------------------------------------------------------------------------

/// Manages replica placement for data items.
#[derive(Debug, Default)]
pub struct ReplicationManager {
    /// Map from data key → list of node IDs holding replicas.
    replicas: HashMap<String, Vec<u64>>,
}

impl ReplicationManager {
    /// Create a new replication manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replicate data to `factor` nodes selected from `nodes`.
    ///
    /// Selection is deterministic: the first `factor` nodes from the slice are
    /// used. The mapping is stored internally and the selected node IDs are
    /// returned.
    ///
    /// If `nodes` has fewer entries than `factor`, all provided nodes are used.
    pub fn replicate(&mut self, data: &[u8], factor: u32, nodes: &[u64]) -> Vec<u64> {
        let count = (factor as usize).min(nodes.len());
        let selected: Vec<u64> = nodes[..count].to_vec();

        // Use the FNV-1a hash of the data as the key for deduplication.
        let key = format!("{:x}", fnv1a_hash(data));
        self.replicas.insert(key, selected.clone());

        selected
    }

    /// Return the nodes holding replicas for the given data.
    #[must_use]
    pub fn replica_nodes(&self, data: &[u8]) -> Option<&[u64]> {
        let key = format!("{:x}", fnv1a_hash(data));
        self.replicas.get(&key).map(|v| v.as_slice())
    }

    /// Total number of tracked replication records.
    #[must_use]
    pub fn record_count(&self) -> usize {
        self.replicas.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── RaftNode ─────────────────────────────────────────────────────────

    #[test]
    fn test_raft_vote_granted_when_term_greater() {
        let node = RaftNode::new(1);
        let resp = node.request_vote(5, 2, 0, 0);
        assert!(resp.vote_granted, "should grant vote for higher term");
        assert_eq!(resp.term, 5);
    }

    #[test]
    fn test_raft_vote_denied_stale_term() {
        let node = RaftNode::new(1);
        // First grant vote at term 5
        node.request_vote(5, 2, 0, 0);
        // Now deny a request with term 3 (stale)
        let resp = node.request_vote(3, 3, 0, 0);
        assert!(!resp.vote_granted, "should deny vote for stale term");
    }

    #[test]
    fn test_raft_vote_denied_already_voted() {
        let node = RaftNode::new(1);
        node.request_vote(1, 2, 0, 0); // vote for node 2
        let resp = node.request_vote(1, 3, 0, 0); // try to vote for node 3 in same term
        assert!(!resp.vote_granted, "should deny double-vote in same term");
    }

    #[test]
    fn test_raft_vote_same_candidate_ok() {
        let node = RaftNode::new(1);
        node.request_vote(1, 2, 0, 0); // vote for node 2
        let resp = node.request_vote(1, 2, 0, 0); // same candidate again
        assert!(
            resp.vote_granted,
            "idempotent vote for same candidate should succeed"
        );
    }

    #[test]
    fn test_raft_vote_new_term_clears_old_vote() {
        let node = RaftNode::new(1);
        node.request_vote(1, 2, 0, 0); // vote for 2 in term 1
        let resp = node.request_vote(2, 3, 0, 0); // new term → vote for 3
        assert!(resp.vote_granted);
        assert_eq!(node.voted_for(), Some(3));
    }

    // ── WorkStealingQueue ─────────────────────────────────────────────────

    #[test]
    fn test_wsq_push_pop_lifo() {
        let mut q: WorkStealingQueue<u32> = WorkStealingQueue::new();
        q.push(1);
        q.push(2);
        q.push(3);
        assert_eq!(q.pop(), Some(3)); // LIFO
        assert_eq!(q.pop(), Some(2));
        assert_eq!(q.pop(), Some(1));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn test_wsq_steal_fifo() {
        let mut q: WorkStealingQueue<u32> = WorkStealingQueue::new();
        q.push(1);
        q.push(2);
        q.push(3);
        assert_eq!(q.steal(), Some(1)); // FIFO
        assert_eq!(q.steal(), Some(2));
        assert_eq!(q.steal(), Some(3));
        assert_eq!(q.steal(), None);
    }

    #[test]
    fn test_wsq_len_and_empty() {
        let mut q: WorkStealingQueue<&str> = WorkStealingQueue::new();
        assert!(q.is_empty());
        q.push("a");
        q.push("b");
        assert_eq!(q.len(), 2);
    }

    // ── BackpressureController ────────────────────────────────────────────

    #[test]
    fn test_backpressure_allows_up_to_max() {
        let bp = BackpressureController::new(3);
        assert!(bp.try_submit());
        assert!(bp.try_submit());
        assert!(bp.try_submit());
        assert!(!bp.try_submit(), "should be rejected when at max");
    }

    #[test]
    fn test_backpressure_complete_frees_slot() {
        let bp = BackpressureController::new(1);
        assert!(bp.try_submit());
        assert!(!bp.try_submit()); // full
        bp.complete_one();
        assert!(bp.try_submit()); // slot freed
    }

    #[test]
    fn test_backpressure_pending_count() {
        let bp = BackpressureController::new(10);
        bp.try_submit();
        bp.try_submit();
        assert_eq!(bp.pending_count(), 2);
        bp.complete_one();
        assert_eq!(bp.pending_count(), 1);
    }

    // ── CheckpointCoordinator ─────────────────────────────────────────────

    #[test]
    fn test_checkpoint_sequence_increments() {
        let mut coord = CheckpointCoordinator::new();
        let s1 = coord.take_checkpoint(1, b"state_a");
        let s2 = coord.take_checkpoint(1, b"state_b");
        assert_eq!(s1, 1);
        assert_eq!(s2, 2);
    }

    #[test]
    fn test_checkpoint_latest() {
        let mut coord = CheckpointCoordinator::new();
        coord.take_checkpoint(1, b"old");
        coord.take_checkpoint(1, b"new");
        let latest = coord.latest_checkpoint(1).expect("should have checkpoint");
        assert_eq!(latest.state, b"new");
        assert_eq!(latest.sequence, 2);
    }

    #[test]
    fn test_checkpoint_independent_per_node() {
        let mut coord = CheckpointCoordinator::new();
        let s1 = coord.take_checkpoint(1, b"n1");
        let s2 = coord.take_checkpoint(2, b"n2");
        assert_eq!(s1, 1);
        assert_eq!(s2, 1); // each node starts at 1
        assert_eq!(coord.checkpoint_count(), 2);
    }

    // ── ConsistentHashRing ────────────────────────────────────────────────

    #[test]
    fn test_hash_ring_get_node_returns_same_for_same_key() {
        let mut ring = ConsistentHashRing::new(100);
        ring.add_node(1);
        ring.add_node(2);
        ring.add_node(3);
        let n1 = ring.get_node(b"my-key");
        let n2 = ring.get_node(b"my-key");
        assert_eq!(n1, n2, "same key should always map to same node");
    }

    #[test]
    fn test_hash_ring_empty_returns_none() {
        let ring = ConsistentHashRing::new(50);
        assert!(ring.get_node(b"anything").is_none());
    }

    #[test]
    fn test_hash_ring_single_node_owns_all() {
        let mut ring = ConsistentHashRing::new(10);
        ring.add_node(42);
        assert_eq!(ring.get_node(b"k1"), Some(42));
        assert_eq!(ring.get_node(b"k2"), Some(42));
    }

    #[test]
    fn test_hash_ring_remove_node() {
        let mut ring = ConsistentHashRing::new(10);
        ring.add_node(1);
        ring.add_node(2);
        ring.remove_node(1);
        assert_eq!(ring.node_count(), 1);
        assert_eq!(ring.get_node(b"any"), Some(2));
    }

    // ── DistributedCircuitBreaker ─────────────────────────────────────────

    #[test]
    fn test_circuit_breaker_opens_after_threshold() {
        let cb = DistributedCircuitBreaker::new(3, 60_000);
        cb.call_failed();
        assert!(!cb.is_open());
        cb.call_failed();
        assert!(!cb.is_open());
        cb.call_failed(); // threshold reached
        assert!(cb.is_open());
    }

    #[test]
    fn test_circuit_breaker_reset() {
        let cb = DistributedCircuitBreaker::new(1, 60_000);
        cb.call_failed();
        assert!(cb.is_open());
        cb.reset();
        assert!(!cb.is_open());
    }

    #[test]
    fn test_circuit_breaker_success_resets_count() {
        let cb = DistributedCircuitBreaker::new(3, 60_000);
        cb.call_failed();
        cb.call_succeeded(); // resets count
        cb.call_failed();
        assert!(!cb.is_open()); // only 1 failure after reset, not yet at 3
    }

    // ── SimpleShardMap ────────────────────────────────────────────────────

    #[test]
    fn test_shard_map_insert_get() {
        let mut sm = SimpleShardMap::new(4);
        sm.insert(b"key1", b"value1".to_vec());
        assert_eq!(sm.get(b"key1"), Some(b"value1".as_slice()));
        assert_eq!(sm.get(b"missing"), None);
    }

    #[test]
    fn test_shard_map_uniform_distribution() {
        let num_shards = 8u32;
        let mut sm = SimpleShardMap::new(num_shards);

        // Insert 1000 keys
        for i in 0u32..1000 {
            let key = i.to_le_bytes();
            sm.insert(&key, key.to_vec());
        }

        let counts = sm.shard_counts();
        let max = *counts.iter().max().expect("should have max");
        let min = *counts.iter().min().expect("should have min");
        // With FNV-1a and 1000 keys, max/min ratio should be within 2x
        assert!(
            max <= min * 2 + 1,
            "distribution too uneven: max={max} min={min}"
        );
    }

    #[test]
    fn test_shard_assign_basic() {
        assert_eq!(shard_assign(0, 4), 0);
        assert_eq!(shard_assign(4, 4), 0);
        assert_eq!(shard_assign(5, 4), 1);
        assert_eq!(shard_assign(7, 4), 3);
    }

    // ── ServiceRegistry ───────────────────────────────────────────────────

    #[test]
    fn test_service_registry_register_and_discover() {
        let reg = ServiceRegistry::with_default_ttl();
        reg.register(1, "10.0.0.1:50052");
        assert_eq!(reg.discover(1), Some("10.0.0.1:50052".to_string()));
    }

    #[test]
    fn test_service_registry_missing_returns_none() {
        let reg = ServiceRegistry::with_default_ttl();
        assert!(reg.discover(99).is_none());
    }

    #[test]
    fn test_service_registry_expired() {
        // TTL of 1 nanosecond so it expires immediately.
        let reg = ServiceRegistry::new(Duration::from_nanos(1));
        reg.register(1, "10.0.0.1:50052");
        // Spin until the entry expires (should be near-instant).
        std::thread::sleep(Duration::from_millis(2));
        assert!(
            reg.discover(1).is_none(),
            "expired entry should return None"
        );
    }

    // ── ReplicationManager ────────────────────────────────────────────────

    #[test]
    fn test_replication_manager_selects_factor_nodes() {
        let mut rm = ReplicationManager::new();
        let nodes = [1u64, 2, 3, 4, 5];
        let selected = rm.replicate(b"my-data", 3, &nodes);
        assert_eq!(selected.len(), 3);
        assert_eq!(selected, vec![1, 2, 3]);
    }

    #[test]
    fn test_replication_manager_fewer_nodes_than_factor() {
        let mut rm = ReplicationManager::new();
        let nodes = [1u64, 2];
        let selected = rm.replicate(b"data", 5, &nodes);
        assert_eq!(selected.len(), 2, "should use all available nodes");
    }

    #[test]
    fn test_replication_manager_lookup() {
        let mut rm = ReplicationManager::new();
        let nodes = [10u64, 20, 30];
        rm.replicate(b"key-data", 2, &nodes);
        let replicas = rm.replica_nodes(b"key-data").expect("should have replicas");
        assert_eq!(replicas, [10, 20]);
    }
}
