//! Two-phase commit (2PC) coordinator.
//!
//! Implements the classic two-phase commit protocol for distributed
//! atomic transactions.  The coordinator drives the prepare and commit/abort
//! phases across a set of named participants.

use std::collections::{HashMap, HashSet};

/// Phase of the two-phase commit protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TwoPhaseState {
    /// Initial state; no transaction in progress.
    Idle,
    /// Prepare messages have been sent; awaiting votes.
    Preparing,
    /// All participants voted yes; commit in progress.
    Committing,
    /// At least one participant voted no, or abort was requested.
    Aborting,
    /// All participants acknowledged the commit.
    Committed,
    /// All participants acknowledged the abort.
    Aborted,
}

/// The coordinator for a single two-phase commit transaction.
///
/// Tracks participant votes and drives the protocol from prepare through
/// commit or abort.  Participants are identified by arbitrary `u64` IDs.
pub struct TwoPhaseCoordinator {
    /// Current protocol state.
    state: TwoPhaseState,
    /// Set of participant IDs that must vote.
    participants: Vec<u64>,
    /// Votes collected during the prepare phase (true = yes, false = no).
    votes: HashMap<u64, bool>,
    /// Set of participants that acknowledged the final decision.
    acks: HashSet<u64>,
}

impl TwoPhaseCoordinator {
    /// Create a new, idle coordinator with no participants.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: TwoPhaseState::Idle,
            participants: Vec::new(),
            votes: HashMap::new(),
            acks: HashSet::new(),
        }
    }

    /// Return the current protocol state.
    #[must_use]
    pub fn state(&self) -> TwoPhaseState {
        self.state
    }

    /// Broadcast a `prepare` message to all `participants`.
    ///
    /// Transitions from `Idle` → `Preparing`.  In this in-process simulation
    /// all participants immediately vote "yes" unless overridden via
    /// [`Self::record_vote`].  Returns `true` if all simulated immediate votes agree,
    /// `false` if any participant immediately votes "no".
    ///
    /// # Arguments
    ///
    /// * `participants` - Slice of participant IDs to include in this transaction.
    pub fn prepare(&mut self, participants: &[u64]) -> bool {
        self.participants = participants.to_vec();
        self.votes.clear();
        self.acks.clear();

        if participants.is_empty() {
            // No participants → vacuously prepared
            self.state = TwoPhaseState::Preparing;
            return true;
        }

        self.state = TwoPhaseState::Preparing;

        // Simulate immediate "yes" votes from all participants
        for &p in participants {
            self.votes.insert(p, true);
        }

        self.all_voted_yes()
    }

    /// Record a vote from participant `id`.
    ///
    /// Must be called while in the `Preparing` state.  Returns `false` if the
    /// coordinator is not in the `Preparing` state or `id` is not a known
    /// participant.
    pub fn record_vote(&mut self, id: u64, vote: bool) -> bool {
        if self.state != TwoPhaseState::Preparing {
            return false;
        }
        if !self.participants.contains(&id) {
            return false;
        }
        self.votes.insert(id, vote);
        true
    }

    /// Commit the transaction.
    ///
    /// Transitions to `Committing` if all participants voted yes, then
    /// immediately transitions to `Committed` (simulating synchronous acks).
    ///
    /// Returns `true` on success, `false` if the protocol state does not allow
    /// commit (e.g. not all votes are yes).
    pub fn commit(&mut self) -> bool {
        if self.state != TwoPhaseState::Preparing || !self.all_voted_yes() {
            return false;
        }
        self.state = TwoPhaseState::Committing;
        // Simulate all participants acknowledging
        for &p in &self.participants {
            self.acks.insert(p);
        }
        self.state = TwoPhaseState::Committed;
        true
    }

    /// Abort the transaction.
    ///
    /// Can be called from `Preparing` or `Committing`.  Transitions through
    /// `Aborting` → `Aborted`.  Returns `false` if already in a terminal state.
    pub fn abort(&mut self) -> bool {
        match self.state {
            TwoPhaseState::Idle | TwoPhaseState::Preparing | TwoPhaseState::Committing => {
                self.state = TwoPhaseState::Aborting;
                // Simulate all participants acknowledging the abort
                for &p in &self.participants {
                    self.acks.insert(p);
                }
                self.state = TwoPhaseState::Aborted;
                true
            }
            _ => false,
        }
    }

    /// Reset the coordinator to `Idle` so it can be reused.
    pub fn reset(&mut self) {
        self.state = TwoPhaseState::Idle;
        self.participants.clear();
        self.votes.clear();
        self.acks.clear();
    }

    /// Return `true` if all participants in the `votes` map voted yes.
    fn all_voted_yes(&self) -> bool {
        if self.participants.is_empty() {
            return true;
        }
        self.participants
            .iter()
            .all(|p| self.votes.get(p).copied().unwrap_or(false))
    }

    /// Number of yes votes collected so far.
    #[must_use]
    pub fn yes_vote_count(&self) -> usize {
        self.votes.values().filter(|&&v| v).count()
    }

    /// Number of no votes collected so far.
    #[must_use]
    pub fn no_vote_count(&self) -> usize {
        self.votes.values().filter(|&&v| !v).count()
    }
}

impl Default for TwoPhaseCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_state_is_idle() {
        let coord = TwoPhaseCoordinator::new();
        assert_eq!(coord.state(), TwoPhaseState::Idle);
    }

    #[test]
    fn test_prepare_transitions_to_preparing() {
        let mut coord = TwoPhaseCoordinator::new();
        let result = coord.prepare(&[1, 2]);
        assert!(result); // all auto yes votes
        assert_eq!(coord.state(), TwoPhaseState::Preparing);
    }

    #[test]
    fn test_commit_after_prepare_succeeds() {
        let mut coord = TwoPhaseCoordinator::new();
        coord.prepare(&[1, 2, 3]);
        let ok = coord.commit();
        assert!(ok);
        assert_eq!(coord.state(), TwoPhaseState::Committed);
    }

    #[test]
    fn test_abort_after_prepare() {
        let mut coord = TwoPhaseCoordinator::new();
        coord.prepare(&[1, 2]);
        let ok = coord.abort();
        assert!(ok);
        assert_eq!(coord.state(), TwoPhaseState::Aborted);
    }

    #[test]
    fn test_no_vote_prevents_commit() {
        let mut coord = TwoPhaseCoordinator::new();
        coord.prepare(&[1, 2, 3]);
        coord.record_vote(2, false); // participant 2 votes no
        let ok = coord.commit();
        assert!(!ok);
        assert_eq!(coord.state(), TwoPhaseState::Preparing); // unchanged
    }

    #[test]
    fn test_abort_after_no_vote() {
        let mut coord = TwoPhaseCoordinator::new();
        coord.prepare(&[1, 2]);
        coord.record_vote(1, false);
        coord.abort();
        assert_eq!(coord.state(), TwoPhaseState::Aborted);
    }

    #[test]
    fn test_abort_terminal_state_returns_false() {
        let mut coord = TwoPhaseCoordinator::new();
        coord.prepare(&[1]);
        coord.commit();
        assert_eq!(coord.state(), TwoPhaseState::Committed);
        let ok = coord.abort();
        assert!(!ok);
    }

    #[test]
    fn test_reset_allows_reuse() {
        let mut coord = TwoPhaseCoordinator::new();
        coord.prepare(&[1]);
        coord.commit();
        coord.reset();
        assert_eq!(coord.state(), TwoPhaseState::Idle);
        coord.prepare(&[2, 3]);
        assert_eq!(coord.state(), TwoPhaseState::Preparing);
        coord.commit();
        assert_eq!(coord.state(), TwoPhaseState::Committed);
    }

    #[test]
    fn test_record_vote_unknown_participant() {
        let mut coord = TwoPhaseCoordinator::new();
        coord.prepare(&[1]);
        let ok = coord.record_vote(99, true); // not a participant
        assert!(!ok);
    }

    #[test]
    fn test_record_vote_wrong_state() {
        let mut coord = TwoPhaseCoordinator::new();
        let ok = coord.record_vote(1, true); // Idle state
        assert!(!ok);
    }

    #[test]
    fn test_empty_participants_commit() {
        let mut coord = TwoPhaseCoordinator::new();
        coord.prepare(&[]);
        coord.commit();
        assert_eq!(coord.state(), TwoPhaseState::Committed);
    }

    #[test]
    fn test_vote_counts() {
        let mut coord = TwoPhaseCoordinator::new();
        coord.prepare(&[1, 2, 3]);
        coord.record_vote(3, false);
        assert_eq!(coord.yes_vote_count(), 2);
        assert_eq!(coord.no_vote_count(), 1);
    }
}
