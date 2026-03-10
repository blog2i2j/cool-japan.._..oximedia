#![allow(dead_code)]
//! Take management for ADR and recording sessions.
//!
//! Tracks multiple takes of dialogue, Foley, or music recordings. Supports
//! rating, comparison, compositing of best segments, and take selection
//! workflows used in professional audio post-production.

use std::collections::HashMap;

/// Unique identifier for a take.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TakeId(String);

impl TakeId {
    /// Create a new take identifier.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Return the string value.
    pub fn value(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TakeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Rating for a take (1-5 stars).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TakeRating(u8);

impl TakeRating {
    /// Create a rating clamped to 1..=5.
    pub fn new(stars: u8) -> Self {
        Self(stars.clamp(1, 5))
    }

    /// Return the numeric star value.
    pub fn stars(self) -> u8 {
        self.0
    }
}

impl std::fmt::Display for TakeRating {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/5", self.0)
    }
}

/// Status of a take in the selection workflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TakeStatus {
    /// Just recorded, not yet reviewed.
    Recorded,
    /// Reviewed but no decision made.
    Reviewed,
    /// Selected as the chosen take.
    Selected,
    /// Selected as a backup / alt.
    Alternate,
    /// Explicitly rejected.
    Rejected,
}

impl std::fmt::Display for TakeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Recorded => write!(f, "recorded"),
            Self::Reviewed => write!(f, "reviewed"),
            Self::Selected => write!(f, "selected"),
            Self::Alternate => write!(f, "alternate"),
            Self::Rejected => write!(f, "rejected"),
        }
    }
}

/// A single recorded take.
#[derive(Debug, Clone)]
pub struct Take {
    /// Unique identifier.
    pub id: TakeId,
    /// Take number within the cue (1-based).
    pub number: u32,
    /// Associated cue or line identifier.
    pub cue_id: String,
    /// Status in the review pipeline.
    pub status: TakeStatus,
    /// Optional rating.
    pub rating: Option<TakeRating>,
    /// Duration in seconds.
    pub duration_secs: f64,
    /// Sample rate.
    pub sample_rate: u32,
    /// Notes from the director or engineer.
    pub notes: String,
    /// File path to the recorded audio.
    pub file_path: String,
    /// Peak level in dB.
    pub peak_db: f64,
    /// RMS level in dB.
    pub rms_db: f64,
    /// Arbitrary metadata.
    pub metadata: HashMap<String, String>,
}

impl Take {
    /// Create a new take.
    pub fn new(id: impl Into<String>, number: u32, cue_id: impl Into<String>) -> Self {
        Self {
            id: TakeId::new(id),
            number,
            cue_id: cue_id.into(),
            status: TakeStatus::Recorded,
            rating: None,
            duration_secs: 0.0,
            sample_rate: 48000,
            notes: String::new(),
            file_path: String::new(),
            peak_db: -96.0,
            rms_db: -96.0,
            metadata: HashMap::new(),
        }
    }

    /// Set the rating.
    pub fn rate(&mut self, stars: u8) {
        self.rating = Some(TakeRating::new(stars));
    }

    /// Set the status.
    pub fn set_status(&mut self, status: TakeStatus) {
        self.status = status;
    }

    /// Set file path and duration.
    pub fn set_audio(&mut self, path: impl Into<String>, duration_secs: f64) {
        self.file_path = path.into();
        self.duration_secs = duration_secs;
    }

    /// Set levels.
    pub fn set_levels(&mut self, peak_db: f64, rms_db: f64) {
        self.peak_db = peak_db;
        self.rms_db = rms_db;
    }

    /// Add a note.
    pub fn add_note(&mut self, note: &str) {
        if !self.notes.is_empty() {
            self.notes.push_str("; ");
        }
        self.notes.push_str(note);
    }

    /// Check if the take is usable (selected or alternate).
    pub fn is_usable(&self) -> bool {
        matches!(self.status, TakeStatus::Selected | TakeStatus::Alternate)
    }
}

/// Manages all takes for a recording session.
#[derive(Debug)]
pub struct TakeManager {
    /// Session identifier.
    pub session_id: String,
    /// All takes, keyed by their id.
    takes: HashMap<TakeId, Take>,
    /// Takes grouped by cue id.
    cue_index: HashMap<String, Vec<TakeId>>,
}

impl TakeManager {
    /// Create a new take manager for a session.
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            takes: HashMap::new(),
            cue_index: HashMap::new(),
        }
    }

    /// Add a take.
    pub fn add_take(&mut self, take: Take) {
        let cue_id = take.cue_id.clone();
        let take_id = take.id.clone();
        self.takes.insert(take_id.clone(), take);
        self.cue_index.entry(cue_id).or_default().push(take_id);
    }

    /// Get a take by id.
    pub fn get(&self, id: &TakeId) -> Option<&Take> {
        self.takes.get(id)
    }

    /// Get a mutable reference to a take by id.
    pub fn get_mut(&mut self, id: &TakeId) -> Option<&mut Take> {
        self.takes.get_mut(id)
    }

    /// Return total number of takes.
    pub fn take_count(&self) -> usize {
        self.takes.len()
    }

    /// Return number of cues with at least one take.
    pub fn cue_count(&self) -> usize {
        self.cue_index.len()
    }

    /// Get all takes for a specific cue, sorted by take number.
    pub fn takes_for_cue(&self, cue_id: &str) -> Vec<&Take> {
        let mut takes: Vec<&Take> = self
            .cue_index
            .get(cue_id)
            .map(|ids| ids.iter().filter_map(|id| self.takes.get(id)).collect())
            .unwrap_or_default();
        takes.sort_by_key(|t| t.number);
        takes
    }

    /// Get the selected take for a cue (if any).
    pub fn selected_take(&self, cue_id: &str) -> Option<&Take> {
        self.takes_for_cue(cue_id)
            .into_iter()
            .find(|t| t.status == TakeStatus::Selected)
    }

    /// Get the best-rated take for a cue.
    pub fn best_rated_take(&self, cue_id: &str) -> Option<&Take> {
        self.takes_for_cue(cue_id)
            .into_iter()
            .filter(|t| t.rating.is_some())
            .max_by_key(|t| t.rating)
    }

    /// Select the specified take (and un-select any other selected take for the same cue).
    pub fn select_take(&mut self, take_id: &TakeId) -> bool {
        let cue_id = match self.takes.get(take_id) {
            Some(t) => t.cue_id.clone(),
            None => return false,
        };

        // Deselect any currently selected take for this cue
        if let Some(ids) = self.cue_index.get(&cue_id) {
            let ids_clone: Vec<TakeId> = ids.clone();
            for id in &ids_clone {
                if let Some(t) = self.takes.get_mut(id) {
                    if t.status == TakeStatus::Selected {
                        t.status = TakeStatus::Reviewed;
                    }
                }
            }
        }

        // Select the requested take
        if let Some(t) = self.takes.get_mut(take_id) {
            t.status = TakeStatus::Selected;
            true
        } else {
            false
        }
    }

    /// Return all takes with a specific status.
    pub fn takes_with_status(&self, status: TakeStatus) -> Vec<&Take> {
        self.takes.values().filter(|t| t.status == status).collect()
    }

    /// Return cues that have no selected take.
    pub fn cues_without_selection(&self) -> Vec<&str> {
        self.cue_index
            .keys()
            .filter(|cue_id| self.selected_take(cue_id).is_none())
            .map(|s| s.as_str())
            .collect()
    }

    /// Compute session statistics.
    #[allow(clippy::cast_precision_loss)]
    pub fn stats(&self) -> TakeManagerStats {
        let total = self.takes.len();
        let selected = self
            .takes
            .values()
            .filter(|t| t.status == TakeStatus::Selected)
            .count();
        let rejected = self
            .takes
            .values()
            .filter(|t| t.status == TakeStatus::Rejected)
            .count();
        let rated = self.takes.values().filter(|t| t.rating.is_some()).count();
        let avg_rating = if rated > 0 {
            let sum: u64 = self
                .takes
                .values()
                .filter_map(|t| t.rating.map(|r| u64::from(r.stars())))
                .sum();
            sum as f64 / rated as f64
        } else {
            0.0
        };
        let total_duration: f64 = self.takes.values().map(|t| t.duration_secs).sum();

        TakeManagerStats {
            total_takes: total,
            selected_count: selected,
            rejected_count: rejected,
            rated_count: rated,
            average_rating: avg_rating,
            total_duration_secs: total_duration,
            cue_count: self.cue_index.len(),
        }
    }
}

/// Summary statistics for a take manager session.
#[derive(Debug, Clone)]
pub struct TakeManagerStats {
    /// Total number of takes.
    pub total_takes: usize,
    /// Number of selected takes.
    pub selected_count: usize,
    /// Number of rejected takes.
    pub rejected_count: usize,
    /// Number of rated takes.
    pub rated_count: usize,
    /// Average rating across rated takes.
    pub average_rating: f64,
    /// Total duration of all takes in seconds.
    pub total_duration_secs: f64,
    /// Number of distinct cues.
    pub cue_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_take(id: &str, number: u32, cue: &str) -> Take {
        let mut t = Take::new(id, number, cue);
        t.set_audio(format!("/audio/{id}.wav"), 5.0);
        t
    }

    #[test]
    fn test_take_id_display() {
        let id = TakeId::new("take-42");
        assert_eq!(format!("{id}"), "take-42");
        assert_eq!(id.value(), "take-42");
    }

    #[test]
    fn test_take_rating_clamping() {
        assert_eq!(TakeRating::new(0).stars(), 1);
        assert_eq!(TakeRating::new(3).stars(), 3);
        assert_eq!(TakeRating::new(10).stars(), 5);
    }

    #[test]
    fn test_take_rating_display() {
        assert_eq!(format!("{}", TakeRating::new(4)), "4/5");
    }

    #[test]
    fn test_take_status_display() {
        assert_eq!(format!("{}", TakeStatus::Recorded), "recorded");
        assert_eq!(format!("{}", TakeStatus::Selected), "selected");
        assert_eq!(format!("{}", TakeStatus::Rejected), "rejected");
    }

    #[test]
    fn test_take_new() {
        let t = Take::new("t1", 1, "cue-1");
        assert_eq!(t.id, TakeId::new("t1"));
        assert_eq!(t.number, 1);
        assert_eq!(t.status, TakeStatus::Recorded);
        assert!(t.rating.is_none());
    }

    #[test]
    fn test_take_rate_and_notes() {
        let mut t = Take::new("t1", 1, "cue-1");
        t.rate(4);
        assert_eq!(t.rating.expect("rating should be valid").stars(), 4);
        t.add_note("good timing");
        t.add_note("slightly off pitch");
        assert!(t.notes.contains("good timing"));
        assert!(t.notes.contains("slightly off pitch"));
    }

    #[test]
    fn test_take_is_usable() {
        let mut t = Take::new("t1", 1, "cue-1");
        assert!(!t.is_usable());
        t.set_status(TakeStatus::Selected);
        assert!(t.is_usable());
        t.set_status(TakeStatus::Alternate);
        assert!(t.is_usable());
        t.set_status(TakeStatus::Rejected);
        assert!(!t.is_usable());
    }

    #[test]
    fn test_manager_add_and_count() {
        let mut mgr = TakeManager::new("session-1");
        mgr.add_take(make_take("t1", 1, "cue-1"));
        mgr.add_take(make_take("t2", 2, "cue-1"));
        mgr.add_take(make_take("t3", 1, "cue-2"));
        assert_eq!(mgr.take_count(), 3);
        assert_eq!(mgr.cue_count(), 2);
    }

    #[test]
    fn test_manager_takes_for_cue() {
        let mut mgr = TakeManager::new("session-1");
        mgr.add_take(make_take("t2", 2, "cue-1"));
        mgr.add_take(make_take("t1", 1, "cue-1"));
        let takes = mgr.takes_for_cue("cue-1");
        assert_eq!(takes.len(), 2);
        // Should be sorted by number
        assert_eq!(takes[0].number, 1);
        assert_eq!(takes[1].number, 2);
    }

    #[test]
    fn test_manager_select_take() {
        let mut mgr = TakeManager::new("session-1");
        mgr.add_take(make_take("t1", 1, "cue-1"));
        mgr.add_take(make_take("t2", 2, "cue-1"));

        assert!(mgr.select_take(&TakeId::new("t1")));
        assert_eq!(
            mgr.get(&TakeId::new("t1"))
                .expect("failed to get value")
                .status,
            TakeStatus::Selected
        );

        // Selecting t2 should deselect t1
        assert!(mgr.select_take(&TakeId::new("t2")));
        assert_eq!(
            mgr.get(&TakeId::new("t1"))
                .expect("failed to get value")
                .status,
            TakeStatus::Reviewed
        );
        assert_eq!(
            mgr.get(&TakeId::new("t2"))
                .expect("failed to get value")
                .status,
            TakeStatus::Selected
        );
    }

    #[test]
    fn test_manager_select_nonexistent() {
        let mut mgr = TakeManager::new("session-1");
        assert!(!mgr.select_take(&TakeId::new("nope")));
    }

    #[test]
    fn test_manager_selected_take() {
        let mut mgr = TakeManager::new("session-1");
        mgr.add_take(make_take("t1", 1, "cue-1"));
        assert!(mgr.selected_take("cue-1").is_none());
        mgr.select_take(&TakeId::new("t1"));
        assert!(mgr.selected_take("cue-1").is_some());
    }

    #[test]
    fn test_manager_best_rated_take() {
        let mut mgr = TakeManager::new("session-1");
        let mut t1 = make_take("t1", 1, "cue-1");
        t1.rate(3);
        let mut t2 = make_take("t2", 2, "cue-1");
        t2.rate(5);
        mgr.add_take(t1);
        mgr.add_take(t2);

        let best = mgr
            .best_rated_take("cue-1")
            .expect("best_rated_take should succeed");
        assert_eq!(best.rating.expect("rating should be valid").stars(), 5);
    }

    #[test]
    fn test_manager_cues_without_selection() {
        let mut mgr = TakeManager::new("session-1");
        mgr.add_take(make_take("t1", 1, "cue-1"));
        mgr.add_take(make_take("t2", 1, "cue-2"));
        mgr.select_take(&TakeId::new("t1"));

        let unselected = mgr.cues_without_selection();
        assert_eq!(unselected.len(), 1);
        assert_eq!(unselected[0], "cue-2");
    }

    #[test]
    fn test_manager_stats() {
        let mut mgr = TakeManager::new("session-1");
        let mut t1 = make_take("t1", 1, "cue-1");
        t1.rate(4);
        t1.set_status(TakeStatus::Selected);
        let mut t2 = make_take("t2", 2, "cue-1");
        t2.rate(2);
        t2.set_status(TakeStatus::Rejected);
        mgr.add_take(t1);
        mgr.add_take(t2);

        let stats = mgr.stats();
        assert_eq!(stats.total_takes, 2);
        assert_eq!(stats.selected_count, 1);
        assert_eq!(stats.rejected_count, 1);
        assert_eq!(stats.rated_count, 2);
        assert!((stats.average_rating - 3.0).abs() < f64::EPSILON);
        assert!((stats.total_duration_secs - 10.0).abs() < f64::EPSILON);
    }
}
