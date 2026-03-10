//! Distributed consensus module (Raft-inspired).
//!
//! Provides a simplified Raft consensus implementation for leader election
//! and distributed log replication in the distributed encoding cluster.

#![allow(dead_code)]

/// Unique identifier for a Raft node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct NodeId(pub u64);

impl NodeId {
    /// Create a new `NodeId`.
    #[must_use]
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get the inner u64 value.
    #[must_use]
    pub fn inner(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node({})", self.0)
    }
}

/// Raft term number.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    serde::Serialize,
    serde::Deserialize,
    Default,
)]
pub struct RaftTerm(pub u64);

impl RaftTerm {
    /// Create a new `RaftTerm`.
    #[must_use]
    pub fn new(term: u64) -> Self {
        Self(term)
    }

    /// Get the inner u64 value.
    #[must_use]
    pub fn inner(self) -> u64 {
        self.0
    }

    /// Increment the term.
    #[must_use]
    pub fn increment(self) -> Self {
        Self(self.0 + 1)
    }
}

impl std::fmt::Display for RaftTerm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Term({})", self.0)
    }
}

/// Role of a Raft node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaftRole {
    /// Follower - the default state.
    Follower,
    /// Candidate - seeking election.
    Candidate,
    /// Leader - coordinating the cluster.
    Leader,
}

impl RaftRole {
    /// Returns true if the node can vote in leader elections.
    #[must_use]
    pub fn can_vote(self) -> bool {
        matches!(self, RaftRole::Follower | RaftRole::Candidate)
    }
}

impl std::fmt::Display for RaftRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RaftRole::Follower => write!(f, "Follower"),
            RaftRole::Candidate => write!(f, "Candidate"),
            RaftRole::Leader => write!(f, "Leader"),
        }
    }
}

/// An entry in the Raft log.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LogEntry {
    /// The term when this entry was created.
    pub term: RaftTerm,
    /// Index of this entry in the log (1-based).
    pub index: u64,
    /// Serialized command (e.g., JSON-encoded operation).
    pub command: String,
}

impl LogEntry {
    /// Create a new log entry.
    pub fn new(term: RaftTerm, index: u64, command: impl Into<String>) -> Self {
        Self {
            term,
            index,
            command: command.into(),
        }
    }
}

/// The Raft replicated log.
#[derive(Debug, Default)]
pub struct RaftLog {
    /// All log entries (0-indexed internally, 1-indexed externally).
    pub entries: Vec<LogEntry>,
}

impl RaftLog {
    /// Create a new empty Raft log.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Append a new entry to the log.
    pub fn append(&mut self, entry: LogEntry) {
        self.entries.push(entry);
    }

    /// Get the entry at the given 1-based index.
    #[must_use]
    pub fn entry_at(&self, idx: u64) -> Option<&LogEntry> {
        if idx == 0 {
            return None;
        }
        self.entries.get((idx - 1) as usize)
    }

    /// Get the index of the last entry (0 if log is empty).
    #[must_use]
    pub fn last_index(&self) -> u64 {
        self.entries.len() as u64
    }

    /// Get the term of the last entry (default term 0 if log is empty).
    #[must_use]
    pub fn last_term(&self) -> RaftTerm {
        self.entries.last().map(|e| e.term).unwrap_or_default()
    }
}

/// A request to vote for a candidate in a Raft election.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VoteRequest {
    /// The candidate's current term.
    pub term: RaftTerm,
    /// The candidate's node ID.
    pub candidate_id: NodeId,
    /// Index of the candidate's last log entry.
    pub last_log_index: u64,
    /// Term of the candidate's last log entry.
    pub last_log_term: RaftTerm,
}

/// A response to a vote request.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VoteResponse {
    /// The current term, for the candidate to update itself.
    pub term: RaftTerm,
    /// True if the vote was granted.
    pub vote_granted: bool,
}

/// A Raft consensus node.
#[derive(Debug)]
pub struct RaftNode {
    /// This node's ID.
    pub id: NodeId,
    /// Current role.
    pub role: RaftRole,
    /// Latest term this node has seen.
    pub current_term: RaftTerm,
    /// `NodeId` this node voted for in the current term.
    pub voted_for: Option<NodeId>,
    /// The replicated log.
    pub log: RaftLog,
    /// Index of highest log entry known to be committed.
    pub commit_index: u64,
    /// Index of highest log entry applied to state machine.
    pub last_applied: u64,
}

impl RaftNode {
    /// Create a new Raft node in Follower state.
    #[must_use]
    pub fn new(id: NodeId) -> Self {
        Self {
            id,
            role: RaftRole::Follower,
            current_term: RaftTerm::default(),
            voted_for: None,
            log: RaftLog::new(),
            commit_index: 0,
            last_applied: 0,
        }
    }

    /// Process an incoming vote request and return a response.
    ///
    /// Grant vote if:
    /// 1. Candidate's term >= our current term.
    /// 2. We haven't voted for anyone else this term.
    /// 3. Candidate's log is at least as up-to-date as ours.
    pub fn process_vote_request(&mut self, req: &VoteRequest) -> VoteResponse {
        // If we see a higher term, update and step down
        if req.term > self.current_term {
            self.step_down(req.term);
        }

        if req.term < self.current_term {
            return VoteResponse {
                term: self.current_term,
                vote_granted: false,
            };
        }

        // Check if we can vote for this candidate
        let already_voted_other = self.voted_for.is_some_and(|v| v != req.candidate_id);

        if already_voted_other {
            return VoteResponse {
                term: self.current_term,
                vote_granted: false,
            };
        }

        // Check log up-to-date-ness
        let our_last_term = self.log.last_term();
        let our_last_index = self.log.last_index();

        let log_ok = req.last_log_term > our_last_term
            || (req.last_log_term == our_last_term && req.last_log_index >= our_last_index);

        if log_ok {
            self.voted_for = Some(req.candidate_id);
            VoteResponse {
                term: self.current_term,
                vote_granted: true,
            }
        } else {
            VoteResponse {
                term: self.current_term,
                vote_granted: false,
            }
        }
    }

    /// Transition this node to Candidate and start a new election.
    pub fn become_candidate(&mut self) {
        self.current_term = self.current_term.increment();
        self.role = RaftRole::Candidate;
        self.voted_for = Some(self.id); // Vote for self
    }

    /// Transition this node to Leader.
    pub fn become_leader(&mut self) {
        self.role = RaftRole::Leader;
    }

    /// Step down to Follower with a new term (e.g., after seeing higher term).
    pub fn step_down(&mut self, new_term: RaftTerm) {
        self.current_term = new_term;
        self.role = RaftRole::Follower;
        self.voted_for = None;
    }
}

/// A timer that tracks election timeouts.
#[derive(Debug, Clone)]
pub struct ElectionTimer {
    /// Timeout duration in milliseconds.
    pub timeout_ms: u64,
    /// Timestamp (ms) of the last reset.
    pub last_reset_ms: u64,
}

impl ElectionTimer {
    /// Create a new election timer.
    #[must_use]
    pub fn new(timeout_ms: u64, now_ms: u64) -> Self {
        Self {
            timeout_ms,
            last_reset_ms: now_ms,
        }
    }

    /// Returns true if the timer has expired at the given time.
    #[must_use]
    pub fn is_expired(&self, now_ms: u64) -> bool {
        now_ms.saturating_sub(self.last_reset_ms) >= self.timeout_ms
    }

    /// Reset the timer to the current time.
    pub fn reset(&mut self, now_ms: u64) {
        self.last_reset_ms = now_ms;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(id: u64) -> RaftNode {
        RaftNode::new(NodeId::new(id))
    }

    #[test]
    fn test_node_initial_state() {
        let node = make_node(1);
        assert_eq!(node.id, NodeId::new(1));
        assert_eq!(node.role, RaftRole::Follower);
        assert_eq!(node.current_term, RaftTerm::default());
        assert!(node.voted_for.is_none());
    }

    #[test]
    fn test_raft_role_can_vote() {
        assert!(RaftRole::Follower.can_vote());
        assert!(RaftRole::Candidate.can_vote());
        assert!(!RaftRole::Leader.can_vote());
    }

    #[test]
    fn test_raft_term_increment() {
        let term = RaftTerm::new(5);
        assert_eq!(term.increment().inner(), 6);
    }

    #[test]
    fn test_raft_log_append_and_entry_at() {
        let mut log = RaftLog::new();
        assert_eq!(log.last_index(), 0);
        assert_eq!(log.last_term(), RaftTerm::default());

        log.append(LogEntry::new(RaftTerm::new(1), 1, "cmd1"));
        log.append(LogEntry::new(RaftTerm::new(1), 2, "cmd2"));

        assert_eq!(log.last_index(), 2);
        assert_eq!(log.last_term(), RaftTerm::new(1));
        assert!(log.entry_at(1).is_some());
        assert_eq!(log.entry_at(1).expect("entry should exist").command, "cmd1");
        assert!(log.entry_at(0).is_none());
        assert!(log.entry_at(3).is_none());
    }

    #[test]
    fn test_become_candidate() {
        let mut node = make_node(1);
        node.become_candidate();
        assert_eq!(node.role, RaftRole::Candidate);
        assert_eq!(node.current_term, RaftTerm::new(1));
        assert_eq!(node.voted_for, Some(NodeId::new(1)));
    }

    #[test]
    fn test_become_leader() {
        let mut node = make_node(1);
        node.become_candidate();
        node.become_leader();
        assert_eq!(node.role, RaftRole::Leader);
    }

    #[test]
    fn test_step_down() {
        let mut node = make_node(1);
        node.become_candidate();
        node.step_down(RaftTerm::new(5));
        assert_eq!(node.role, RaftRole::Follower);
        assert_eq!(node.current_term, RaftTerm::new(5));
        assert!(node.voted_for.is_none());
    }

    #[test]
    fn test_vote_request_grant() {
        let mut node = make_node(2);
        let req = VoteRequest {
            term: RaftTerm::new(1),
            candidate_id: NodeId::new(1),
            last_log_index: 0,
            last_log_term: RaftTerm::default(),
        };
        let resp = node.process_vote_request(&req);
        assert!(resp.vote_granted);
    }

    #[test]
    fn test_vote_request_deny_lower_term() {
        let mut node = make_node(2);
        node.step_down(RaftTerm::new(3));
        let req = VoteRequest {
            term: RaftTerm::new(2),
            candidate_id: NodeId::new(1),
            last_log_index: 0,
            last_log_term: RaftTerm::default(),
        };
        let resp = node.process_vote_request(&req);
        assert!(!resp.vote_granted);
    }

    #[test]
    fn test_vote_request_deny_already_voted() {
        let mut node = make_node(2);
        let req1 = VoteRequest {
            term: RaftTerm::new(1),
            candidate_id: NodeId::new(1),
            last_log_index: 0,
            last_log_term: RaftTerm::default(),
        };
        let req2 = VoteRequest {
            term: RaftTerm::new(1),
            candidate_id: NodeId::new(3),
            last_log_index: 0,
            last_log_term: RaftTerm::default(),
        };
        node.process_vote_request(&req1);
        let resp2 = node.process_vote_request(&req2);
        assert!(!resp2.vote_granted);
    }

    #[test]
    fn test_vote_deny_stale_log() {
        let mut node = make_node(2);
        // Node 2 has log entries at term 2
        node.log.append(LogEntry::new(RaftTerm::new(2), 1, "x"));
        let req = VoteRequest {
            term: RaftTerm::new(3),
            candidate_id: NodeId::new(1),
            last_log_index: 0,
            last_log_term: RaftTerm::new(1), // candidate's log is older
        };
        let resp = node.process_vote_request(&req);
        assert!(!resp.vote_granted);
    }

    #[test]
    fn test_election_timer_expired() {
        let timer = ElectionTimer::new(150, 1000);
        assert!(!timer.is_expired(1100));
        assert!(timer.is_expired(1150));
        assert!(timer.is_expired(1200));
    }

    #[test]
    fn test_election_timer_reset() {
        let mut timer = ElectionTimer::new(150, 1000);
        timer.reset(1100);
        assert!(!timer.is_expired(1200)); // only 100ms elapsed after reset
        assert!(timer.is_expired(1250));
    }

    #[test]
    fn test_node_id_display() {
        let id = NodeId::new(42);
        assert_eq!(format!("{}", id), "Node(42)");
    }

    #[test]
    fn test_raft_term_ordering() {
        assert!(RaftTerm::new(5) > RaftTerm::new(3));
        assert!(RaftTerm::new(1) < RaftTerm::new(2));
        assert_eq!(RaftTerm::new(4), RaftTerm::new(4));
    }
}
