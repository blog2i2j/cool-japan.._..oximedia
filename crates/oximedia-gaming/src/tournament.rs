//! Tournament bracket management and match tracking for competitive gaming streams.
//!
//! Provides [`TournamentFormat`] to describe bracket types, [`MatchResult`]
//! for individual bout outcomes, and [`TournamentBracket`] to manage
//! participants, seeding, and match progression.

#![allow(dead_code)]

use std::collections::HashMap;

/// Supported tournament bracket formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TournamentFormat {
    /// Single-elimination: one loss and you are out.
    SingleElimination,
    /// Double-elimination: must lose twice to be eliminated.
    DoubleElimination,
    /// Round-robin: every participant plays every other participant.
    RoundRobin,
    /// Swiss-system: participants are matched against opponents with similar
    /// records over a fixed number of rounds.
    Swiss,
    /// Free-for-all: all participants compete simultaneously (e.g. battle royale).
    FreeForAll,
}

impl TournamentFormat {
    /// Human-readable name for display in overlays and graphics.
    #[must_use]
    pub fn display_name(self) -> &'static str {
        match self {
            Self::SingleElimination => "Single Elimination",
            Self::DoubleElimination => "Double Elimination",
            Self::RoundRobin => "Round Robin",
            Self::Swiss => "Swiss System",
            Self::FreeForAll => "Free-for-All",
        }
    }

    /// Minimum number of rounds needed for `n` participants in this format.
    ///
    /// Returns `0` when `n <= 1`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn min_rounds(self, n: usize) -> usize {
        if n <= 1 {
            return 0;
        }
        match self {
            Self::SingleElimination => (n as f64).log2().ceil() as usize,
            Self::DoubleElimination => {
                let upper = (n as f64).log2().ceil() as usize;
                // Lower bracket adds roughly the same number again.
                upper * 2
            }
            Self::RoundRobin => n - 1,
            Self::Swiss => ((n as f64).log2().ceil() as usize).max(3),
            Self::FreeForAll => 1,
        }
    }
}

/// Outcome of a single match between two participants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchResult {
    /// Identifier of the first participant.
    pub player_a: String,
    /// Identifier of the second participant.
    pub player_b: String,
    /// Score achieved by player A.
    pub score_a: u32,
    /// Score achieved by player B.
    pub score_b: u32,
    /// Whether the match was decided by a tiebreaker.
    pub tiebreaker: bool,
}

impl MatchResult {
    /// Create a new match result.
    #[must_use]
    pub fn new(
        player_a: impl Into<String>,
        player_b: impl Into<String>,
        score_a: u32,
        score_b: u32,
    ) -> Self {
        Self {
            player_a: player_a.into(),
            player_b: player_b.into(),
            score_a,
            score_b,
            tiebreaker: false,
        }
    }

    /// Mark this result as having been decided by a tiebreaker.
    #[must_use]
    pub fn with_tiebreaker(mut self) -> Self {
        self.tiebreaker = true;
        self
    }

    /// Name of the winner, or `None` if the match is a draw.
    #[must_use]
    pub fn winner(&self) -> Option<&str> {
        if self.score_a > self.score_b {
            Some(&self.player_a)
        } else if self.score_b > self.score_a {
            Some(&self.player_b)
        } else {
            None
        }
    }

    /// Name of the loser, or `None` if the match is a draw.
    #[must_use]
    pub fn loser(&self) -> Option<&str> {
        if self.score_a > self.score_b {
            Some(&self.player_b)
        } else if self.score_b > self.score_a {
            Some(&self.player_a)
        } else {
            None
        }
    }

    /// Whether the match ended in a draw.
    #[must_use]
    pub fn is_draw(&self) -> bool {
        self.score_a == self.score_b
    }

    /// Total combined score across both players.
    #[must_use]
    pub fn total_score(&self) -> u32 {
        self.score_a + self.score_b
    }

    /// Score difference (absolute value).
    #[must_use]
    pub fn score_difference(&self) -> u32 {
        self.score_a.abs_diff(self.score_b)
    }
}

/// An error returned by [`TournamentBracket`] operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TournamentError {
    /// A participant with the same name already exists.
    DuplicateParticipant(String),
    /// The bracket has already been sealed/started; no more edits allowed.
    BracketSealed,
    /// Not enough participants to run this format.
    NotEnoughParticipants {
        /// Number currently registered.
        have: usize,
        /// Minimum required for the format.
        need: usize,
    },
    /// A referenced participant was not found.
    ParticipantNotFound(String),
}

impl std::fmt::Display for TournamentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateParticipant(name) => write!(f, "duplicate participant: {name}"),
            Self::BracketSealed => write!(f, "bracket is sealed"),
            Self::NotEnoughParticipants { have, need } => {
                write!(f, "not enough participants: have {have}, need {need}")
            }
            Self::ParticipantNotFound(name) => write!(f, "participant not found: {name}"),
        }
    }
}

/// Manages a tournament bracket: participants, seedings, and recorded matches.
#[derive(Debug, Clone)]
pub struct TournamentBracket {
    /// Name of the tournament.
    pub name: String,
    /// Format used.
    pub format: TournamentFormat,
    /// Seeded list of participant names (index 0 = seed 1).
    participants: Vec<String>,
    /// Recorded match results in chronological order.
    results: Vec<MatchResult>,
    /// Whether the bracket has been sealed (started).
    sealed: bool,
    /// Win counts per participant name.
    wins: HashMap<String, u32>,
    /// Loss counts per participant name.
    losses: HashMap<String, u32>,
}

impl TournamentBracket {
    /// Create a new, empty bracket.
    #[must_use]
    pub fn new(name: impl Into<String>, format: TournamentFormat) -> Self {
        Self {
            name: name.into(),
            format,
            participants: Vec::new(),
            results: Vec::new(),
            sealed: false,
            wins: HashMap::new(),
            losses: HashMap::new(),
        }
    }

    /// Register a participant.
    ///
    /// # Errors
    ///
    /// Returns [`TournamentError::BracketSealed`] if the bracket has been
    /// sealed, or [`TournamentError::DuplicateParticipant`] if the name
    /// already exists.
    pub fn add_participant(&mut self, name: impl Into<String>) -> Result<(), TournamentError> {
        if self.sealed {
            return Err(TournamentError::BracketSealed);
        }
        let name = name.into();
        if self.participants.contains(&name) {
            return Err(TournamentError::DuplicateParticipant(name));
        }
        self.wins.insert(name.clone(), 0);
        self.losses.insert(name.clone(), 0);
        self.participants.push(name);
        Ok(())
    }

    /// Seal the bracket so that matches can be recorded.
    ///
    /// # Errors
    ///
    /// Returns [`TournamentError::NotEnoughParticipants`] when fewer than 2
    /// participants have been registered.
    pub fn seal(&mut self) -> Result<(), TournamentError> {
        let min = 2;
        if self.participants.len() < min {
            return Err(TournamentError::NotEnoughParticipants {
                have: self.participants.len(),
                need: min,
            });
        }
        self.sealed = true;
        Ok(())
    }

    /// Record a match result.
    ///
    /// # Errors
    ///
    /// Returns an error if the bracket has not been sealed or if either
    /// participant is unknown.
    pub fn record_match(&mut self, result: MatchResult) -> Result<(), TournamentError> {
        if !self.sealed {
            return Err(TournamentError::BracketSealed);
        }
        if !self.participants.contains(&result.player_a) {
            return Err(TournamentError::ParticipantNotFound(
                result.player_a.clone(),
            ));
        }
        if !self.participants.contains(&result.player_b) {
            return Err(TournamentError::ParticipantNotFound(
                result.player_b.clone(),
            ));
        }
        if let Some(w) = result.winner() {
            *self.wins.entry(w.to_owned()).or_default() += 1;
        }
        if let Some(l) = result.loser() {
            *self.losses.entry(l.to_owned()).or_default() += 1;
        }
        self.results.push(result);
        Ok(())
    }

    /// Number of registered participants.
    #[must_use]
    pub fn participant_count(&self) -> usize {
        self.participants.len()
    }

    /// Whether the bracket is sealed.
    #[must_use]
    pub fn is_sealed(&self) -> bool {
        self.sealed
    }

    /// Number of recorded matches.
    #[must_use]
    pub fn match_count(&self) -> usize {
        self.results.len()
    }

    /// All recorded match results.
    #[must_use]
    pub fn results(&self) -> &[MatchResult] {
        &self.results
    }

    /// Win count for a specific participant.
    #[must_use]
    pub fn wins_for(&self, name: &str) -> u32 {
        self.wins.get(name).copied().unwrap_or(0)
    }

    /// Loss count for a specific participant.
    #[must_use]
    pub fn losses_for(&self, name: &str) -> u32 {
        self.losses.get(name).copied().unwrap_or(0)
    }

    /// Participants sorted by wins descending (simple standings).
    #[must_use]
    pub fn standings(&self) -> Vec<(String, u32, u32)> {
        let mut v: Vec<(String, u32, u32)> = self
            .participants
            .iter()
            .map(|p| {
                (
                    p.clone(),
                    self.wins.get(p).copied().unwrap_or(0),
                    self.losses.get(p).copied().unwrap_or(0),
                )
            })
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.2.cmp(&b.2)));
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- TournamentFormat ---

    #[test]
    fn test_format_display_names() {
        assert_eq!(
            TournamentFormat::SingleElimination.display_name(),
            "Single Elimination"
        );
        assert_eq!(
            TournamentFormat::DoubleElimination.display_name(),
            "Double Elimination"
        );
        assert_eq!(TournamentFormat::RoundRobin.display_name(), "Round Robin");
        assert_eq!(TournamentFormat::Swiss.display_name(), "Swiss System");
        assert_eq!(TournamentFormat::FreeForAll.display_name(), "Free-for-All");
    }

    #[test]
    fn test_min_rounds_single_elimination() {
        assert_eq!(TournamentFormat::SingleElimination.min_rounds(8), 3);
        assert_eq!(TournamentFormat::SingleElimination.min_rounds(16), 4);
        assert_eq!(TournamentFormat::SingleElimination.min_rounds(1), 0);
    }

    #[test]
    fn test_min_rounds_round_robin() {
        assert_eq!(TournamentFormat::RoundRobin.min_rounds(4), 3);
        assert_eq!(TournamentFormat::RoundRobin.min_rounds(6), 5);
    }

    #[test]
    fn test_min_rounds_free_for_all() {
        assert_eq!(TournamentFormat::FreeForAll.min_rounds(100), 1);
    }

    // --- MatchResult ---

    #[test]
    fn test_match_result_winner() {
        let m = MatchResult::new("Alice", "Bob", 3, 1);
        assert_eq!(m.winner(), Some("Alice"));
        assert_eq!(m.loser(), Some("Bob"));
        assert!(!m.is_draw());
    }

    #[test]
    fn test_match_result_draw() {
        let m = MatchResult::new("Alice", "Bob", 2, 2);
        assert!(m.is_draw());
        assert!(m.winner().is_none());
        assert!(m.loser().is_none());
    }

    #[test]
    fn test_match_result_tiebreaker() {
        let m = MatchResult::new("X", "Y", 1, 0).with_tiebreaker();
        assert!(m.tiebreaker);
    }

    #[test]
    fn test_match_total_score() {
        let m = MatchResult::new("A", "B", 5, 3);
        assert_eq!(m.total_score(), 8);
    }

    #[test]
    fn test_match_score_difference() {
        let m = MatchResult::new("A", "B", 7, 3);
        assert_eq!(m.score_difference(), 4);
    }

    // --- TournamentBracket ---

    fn make_bracket() -> TournamentBracket {
        let mut b = TournamentBracket::new("Test Cup", TournamentFormat::SingleElimination);
        b.add_participant("Alice")
            .expect("add participant should succeed");
        b.add_participant("Bob")
            .expect("add participant should succeed");
        b.add_participant("Charlie")
            .expect("add participant should succeed");
        b.add_participant("Diana")
            .expect("add participant should succeed");
        b.seal().expect("seal should succeed");
        b
    }

    #[test]
    fn test_bracket_creation() {
        let b = TournamentBracket::new("Cup", TournamentFormat::RoundRobin);
        assert_eq!(b.name, "Cup");
        assert_eq!(b.format, TournamentFormat::RoundRobin);
        assert_eq!(b.participant_count(), 0);
    }

    #[test]
    fn test_add_participants() {
        let mut b = TournamentBracket::new("Cup", TournamentFormat::Swiss);
        b.add_participant("P1")
            .expect("add participant should succeed");
        b.add_participant("P2")
            .expect("add participant should succeed");
        assert_eq!(b.participant_count(), 2);
    }

    #[test]
    fn test_duplicate_participant_error() {
        let mut b = TournamentBracket::new("Cup", TournamentFormat::Swiss);
        b.add_participant("P1")
            .expect("add participant should succeed");
        let err = b.add_participant("P1").unwrap_err();
        assert_eq!(err, TournamentError::DuplicateParticipant("P1".into()));
    }

    #[test]
    fn test_seal_too_few_error() {
        let mut b = TournamentBracket::new("Cup", TournamentFormat::Swiss);
        b.add_participant("Solo")
            .expect("add participant should succeed");
        let err = b.seal().unwrap_err();
        assert!(matches!(err, TournamentError::NotEnoughParticipants { .. }));
    }

    #[test]
    fn test_record_match_and_standings() {
        let mut b = make_bracket();
        b.record_match(MatchResult::new("Alice", "Bob", 3, 1))
            .expect("should succeed");
        b.record_match(MatchResult::new("Charlie", "Diana", 2, 0))
            .expect("should succeed");
        b.record_match(MatchResult::new("Alice", "Charlie", 2, 1))
            .expect("should succeed");

        assert_eq!(b.match_count(), 3);
        assert_eq!(b.wins_for("Alice"), 2);
        assert_eq!(b.losses_for("Bob"), 1);

        let standings = b.standings();
        assert_eq!(standings[0].0, "Alice");
    }

    #[test]
    fn test_record_match_before_seal_fails() {
        let mut b = TournamentBracket::new("Cup", TournamentFormat::Swiss);
        b.add_participant("A")
            .expect("add participant should succeed");
        b.add_participant("B")
            .expect("add participant should succeed");
        let err = b
            .record_match(MatchResult::new("A", "B", 1, 0))
            .unwrap_err();
        assert_eq!(err, TournamentError::BracketSealed);
    }

    #[test]
    fn test_record_match_unknown_participant() {
        let mut b = make_bracket();
        let err = b
            .record_match(MatchResult::new("Alice", "Unknown", 1, 0))
            .unwrap_err();
        assert!(matches!(err, TournamentError::ParticipantNotFound(_)));
    }
}
