//! Automatic playlist generation from a seed item.
//!
//! [`PlaylistGenerator`] builds an ordered playlist by expanding from one or more
//! seed content IDs.  It applies transition scoring (genre overlap, energy delta,
//! tempo compatibility) to produce smooth progressions, enforces a target duration
//! budget, and optionally models energy flow (e.g. low → peak → cool-down arcs).
//!
//! # Architecture
//!
//! ```text
//! Seed items
//!   │
//!   ▼
//! CandidatePool  ──score──►  TransitionScorer  ──sort──►  [ranked candidates]
//!   │                                                             │
//!   │                 EnergyFlow arc constraint                   │
//!   └──────────────────────────────────────────────────────────►  │
//!                                                                  ▼
//!                                              Playlist (ordered, duration-bounded)
//! ```

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use crate::error::{RecommendError, RecommendResult};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Energy level of a content item (0 = lowest, 1 = highest).
pub type Energy = f32;

/// Tempo of a content item in BPM (beats per minute) or an equivalent
/// normalized score for non-music media.
pub type Tempo = f32;

/// Describes the audio/content properties used for transition scoring.
#[derive(Debug, Clone)]
pub struct ContentTrack {
    /// Unique content identifier.
    pub id: u64,
    /// Title (informational).
    pub title: String,
    /// Genre tags (multiple genres supported).
    pub genres: Vec<String>,
    /// Normalized energy in `[0.0, 1.0]`.
    pub energy: Energy,
    /// Tempo (BPM or normalized equivalent).
    pub tempo: Tempo,
    /// Duration in seconds.
    pub duration_s: f64,
    /// Popularity score in `[0.0, 1.0]` (used as tiebreaker).
    pub popularity: f32,
}

impl ContentTrack {
    /// Creates a new [`ContentTrack`] with the given ID and title, defaulting
    /// all numeric fields to zero/empty.
    #[must_use]
    pub fn new(id: u64, title: impl Into<String>) -> Self {
        Self {
            id,
            title: title.into(),
            genres: Vec::new(),
            energy: 0.5,
            tempo: 120.0,
            duration_s: 180.0,
            popularity: 0.5,
        }
    }

    /// Builder: set genres.
    #[must_use]
    pub fn with_genres(mut self, genres: Vec<String>) -> Self {
        self.genres = genres;
        self
    }

    /// Builder: set energy level.
    #[must_use]
    pub fn with_energy(mut self, energy: Energy) -> Self {
        self.energy = energy.clamp(0.0, 1.0);
        self
    }

    /// Builder: set tempo.
    #[must_use]
    pub fn with_tempo(mut self, tempo: Tempo) -> Self {
        self.tempo = tempo.max(0.0);
        self
    }

    /// Builder: set duration.
    #[must_use]
    pub fn with_duration(mut self, duration_s: f64) -> Self {
        self.duration_s = duration_s.max(0.0);
        self
    }

    /// Builder: set popularity.
    #[must_use]
    pub fn with_popularity(mut self, popularity: f32) -> Self {
        self.popularity = popularity.clamp(0.0, 1.0);
        self
    }
}

// ---------------------------------------------------------------------------
// Energy flow arc
// ---------------------------------------------------------------------------

/// Predefined energy flow arcs that shape how energy evolves across the playlist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnergyArc {
    /// Constant energy throughout — no arc applied.
    Flat,
    /// Start low, rise to a peak in the middle, then cool down.
    RisePeak,
    /// Steadily ascending energy towards the end.
    Ascend,
    /// Begin high, gradually descend to low energy.
    Descend,
    /// Alternating high/low/high/low energy pattern.
    Alternating,
}

impl EnergyArc {
    /// Returns the *target* energy for position `pos` in a playlist of length
    /// `total` (both 0-indexed).
    ///
    /// Returns a value in `[0.0, 1.0]`.
    #[must_use]
    pub fn target_energy(self, pos: usize, total: usize) -> f32 {
        if total == 0 {
            return 0.5;
        }
        let t = pos as f32 / total.max(1) as f32; // 0..1
        match self {
            Self::Flat => 0.5,
            Self::RisePeak => {
                // Triangle: rises to 1.0 at t=0.5, falls back to ~0.2
                if t <= 0.5 {
                    0.2 + 1.6 * t
                } else {
                    1.0 - 1.6 * (t - 0.5)
                }
            }
            Self::Ascend => 0.2 + 0.8 * t,
            Self::Descend => 1.0 - 0.8 * t,
            Self::Alternating => {
                if (pos % 2) == 0 {
                    0.7
                } else {
                    0.3
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Transition scorer
// ---------------------------------------------------------------------------

/// Weights used by [`TransitionScorer`].
#[derive(Debug, Clone)]
pub struct TransitionWeights {
    /// Weight for genre overlap component.
    pub genre: f32,
    /// Weight for energy proximity component.
    pub energy: f32,
    /// Weight for tempo compatibility component.
    pub tempo: f32,
    /// Weight for popularity tiebreaker.
    pub popularity: f32,
}

impl Default for TransitionWeights {
    fn default() -> Self {
        Self {
            genre: 0.40,
            energy: 0.30,
            tempo: 0.20,
            popularity: 0.10,
        }
    }
}

/// Computes transition compatibility between two [`ContentTrack`] items.
///
/// All component scores are in `[0.0, 1.0]` and are combined as a weighted
/// sum using [`TransitionWeights`].
pub struct TransitionScorer {
    weights: TransitionWeights,
}

impl TransitionScorer {
    /// Creates a new scorer with default weights.
    #[must_use]
    pub fn new() -> Self {
        Self {
            weights: TransitionWeights::default(),
        }
    }

    /// Creates a scorer with custom weights.
    ///
    /// # Errors
    ///
    /// Returns [`RecommendError::Other`] if any weight is negative.
    pub fn with_weights(weights: TransitionWeights) -> RecommendResult<Self> {
        if weights.genre < 0.0
            || weights.energy < 0.0
            || weights.tempo < 0.0
            || weights.popularity < 0.0
        {
            return Err(RecommendError::Other(
                "TransitionWeights must be non-negative".to_string(),
            ));
        }
        Ok(Self { weights })
    }

    /// Computes a transition score from `from` to `to`.
    ///
    /// A score of `1.0` means a perfect match; `0.0` means no compatibility.
    #[must_use]
    pub fn score(&self, from: &ContentTrack, to: &ContentTrack) -> f32 {
        let genre_score = Self::genre_overlap(from, to);
        let energy_score = 1.0 - (from.energy - to.energy).abs();
        let tempo_score = Self::tempo_compat(from.tempo, to.tempo);
        let w = &self.weights;
        let total_w = w.genre + w.energy + w.tempo + w.popularity;
        if total_w <= 0.0 {
            return 0.0;
        }
        (w.genre * genre_score
            + w.energy * energy_score
            + w.tempo * tempo_score
            + w.popularity * to.popularity)
            / total_w
    }

    /// Scores a candidate given an *energy arc target* for the slot being filled.
    #[must_use]
    pub fn score_with_arc(
        &self,
        from: &ContentTrack,
        to: &ContentTrack,
        arc_target_energy: f32,
    ) -> f32 {
        let base = self.score(from, to);
        let arc_penalty = (to.energy - arc_target_energy).abs();
        (base * 0.8 + (1.0 - arc_penalty) * 0.2).clamp(0.0, 1.0)
    }

    // -- private helpers --

    fn genre_overlap(a: &ContentTrack, b: &ContentTrack) -> f32 {
        if a.genres.is_empty() || b.genres.is_empty() {
            return 0.0;
        }
        let set_a: HashSet<&str> = a.genres.iter().map(String::as_str).collect();
        let set_b: HashSet<&str> = b.genres.iter().map(String::as_str).collect();
        let intersection = set_a.intersection(&set_b).count();
        let union = set_a.union(&set_b).count();
        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    fn tempo_compat(bpm_a: f32, bpm_b: f32) -> f32 {
        // Consider tracks within ±10 BPM as fully compatible; fall off linearly.
        let diff = (bpm_a - bpm_b).abs();
        (1.0 - diff / 60.0).clamp(0.0, 1.0)
    }
}

impl Default for TransitionScorer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Playlist configuration
// ---------------------------------------------------------------------------

/// Configuration for playlist generation.
#[derive(Debug, Clone)]
pub struct PlaylistConfig {
    /// Maximum total duration of the playlist in seconds.
    /// `None` means no duration cap.
    pub max_duration_s: Option<f64>,
    /// Minimum total duration of the playlist in seconds.
    pub min_duration_s: f64,
    /// Maximum number of tracks regardless of duration.
    pub max_tracks: usize,
    /// Energy arc to apply.
    pub energy_arc: EnergyArc,
    /// Weights for transition scoring.
    pub transition_weights: TransitionWeights,
    /// Disallow consecutive tracks from the same genre.
    pub avoid_genre_repeats: bool,
    /// Minimum transition score to include a candidate (0 = accept all).
    pub min_transition_score: f32,
}

impl Default for PlaylistConfig {
    fn default() -> Self {
        Self {
            max_duration_s: Some(60.0 * 60.0), // 1 hour
            min_duration_s: 0.0,
            max_tracks: 30,
            energy_arc: EnergyArc::Flat,
            transition_weights: TransitionWeights::default(),
            avoid_genre_repeats: false,
            min_transition_score: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// PlaylistGenerator
// ---------------------------------------------------------------------------

/// Builds playlists by iteratively picking the best-scoring next track from a
/// candidate pool.
///
/// # Algorithm
///
/// 1. Add seed tracks to the playlist unconditionally.
/// 2. Score every remaining candidate against the *last track added*, adjusted
///    for the arc-target energy at the next slot.
/// 3. Pick the highest-scoring candidate that meets `min_transition_score`.
/// 4. Repeat until `max_tracks` or duration budget is exhausted.
pub struct PlaylistGenerator {
    /// Internal pool of available tracks indexed by ID.
    pool: HashMap<u64, ContentTrack>,
    /// Transition scorer.
    scorer: TransitionScorer,
}

impl PlaylistGenerator {
    /// Creates an empty [`PlaylistGenerator`] with default transition weights.
    #[must_use]
    pub fn new() -> Self {
        Self {
            pool: HashMap::new(),
            scorer: TransitionScorer::new(),
        }
    }

    /// Adds a single track to the candidate pool.
    pub fn add_track(&mut self, track: ContentTrack) {
        self.pool.insert(track.id, track);
    }

    /// Adds multiple tracks to the candidate pool.
    pub fn add_tracks(&mut self, tracks: impl IntoIterator<Item = ContentTrack>) {
        for t in tracks {
            self.pool.insert(t.id, t);
        }
    }

    /// Returns the number of tracks currently in the pool.
    #[must_use]
    pub fn pool_size(&self) -> usize {
        self.pool.len()
    }

    /// Generates a playlist from `seed_ids`.
    ///
    /// Seed IDs that are not found in the pool are silently skipped.
    ///
    /// # Errors
    ///
    /// Returns [`RecommendError::InsufficientData`] when the pool contains no
    /// tracks and no seeds can be resolved.
    pub fn generate(
        &self,
        seed_ids: &[u64],
        config: &PlaylistConfig,
    ) -> RecommendResult<Vec<ContentTrack>> {
        if self.pool.is_empty() {
            return Err(RecommendError::InsufficientData(
                "Candidate pool is empty".to_string(),
            ));
        }

        let scorer = TransitionScorer::with_weights(config.transition_weights.clone())?;
        let mut playlist: Vec<ContentTrack> = Vec::new();
        let mut used_ids: HashSet<u64> = HashSet::new();
        let mut total_duration = 0.0_f64;

        // --- Phase 1: add seeds ---
        for &sid in seed_ids {
            if used_ids.contains(&sid) {
                continue;
            }
            if let Some(track) = self.pool.get(&sid) {
                let new_dur = total_duration + track.duration_s;
                if let Some(max) = config.max_duration_s {
                    if new_dur > max && !playlist.is_empty() {
                        break;
                    }
                }
                if playlist.len() >= config.max_tracks {
                    break;
                }
                used_ids.insert(track.id);
                total_duration += track.duration_s;
                playlist.push(track.clone());
            }
        }

        // If no seeds were resolved, seed with the highest-popularity track.
        if playlist.is_empty() {
            if let Some(best) = self.pool.values().max_by(|a, b| {
                a.popularity
                    .partial_cmp(&b.popularity)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                used_ids.insert(best.id);
                total_duration += best.duration_s;
                playlist.push(best.clone());
            }
        }

        // --- Phase 2: expand ---
        let estimated_total = config.max_tracks;

        while playlist.len() < config.max_tracks {
            // Check duration budget
            if let Some(max) = config.max_duration_s {
                if total_duration >= max {
                    break;
                }
            }

            let last = match playlist.last() {
                Some(t) => t,
                None => break,
            };
            let pos = playlist.len();
            let arc_target = config.energy_arc.target_energy(pos, estimated_total);

            // Score all unused candidates
            let mut best_score = -1.0_f32;
            let mut best_id: Option<u64> = None;

            for candidate in self.pool.values() {
                if used_ids.contains(&candidate.id) {
                    continue;
                }
                // Duration budget check
                if let Some(max) = config.max_duration_s {
                    if total_duration + candidate.duration_s > max {
                        continue;
                    }
                }
                // Genre repeat avoidance
                if config.avoid_genre_repeats && !last.genres.is_empty() {
                    if !candidate.genres.is_empty()
                        && candidate.genres.iter().any(|g| last.genres.contains(g))
                    {
                        continue;
                    }
                }
                let s = scorer.score_with_arc(last, candidate, arc_target);
                if s < config.min_transition_score {
                    continue;
                }
                if s > best_score {
                    best_score = s;
                    best_id = Some(candidate.id);
                }
            }

            match best_id {
                Some(id) => {
                    let track = self.pool[&id].clone();
                    total_duration += track.duration_s;
                    used_ids.insert(id);
                    playlist.push(track);
                }
                None => break, // no more suitable candidates
            }
        }

        // Enforce minimum duration — if we can't meet it, return what we have
        // (caller may decide whether that is acceptable).
        if total_duration < config.min_duration_s && playlist.is_empty() {
            return Err(RecommendError::InsufficientData(
                "Could not meet minimum playlist duration".to_string(),
            ));
        }

        Ok(playlist)
    }
}

impl Default for PlaylistGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Playlist stats helper
// ---------------------------------------------------------------------------

/// Summarises a generated playlist.
#[derive(Debug, Clone)]
pub struct PlaylistStats {
    /// Number of tracks.
    pub track_count: usize,
    /// Total duration in seconds.
    pub total_duration_s: f64,
    /// Average energy.
    pub avg_energy: f32,
    /// Average transition score (consecutive pairs).
    pub avg_transition_score: f32,
    /// Distinct genres present.
    pub distinct_genres: usize,
}

impl PlaylistStats {
    /// Computes statistics for the given playlist.
    #[must_use]
    pub fn compute(playlist: &[ContentTrack]) -> Self {
        let track_count = playlist.len();
        if track_count == 0 {
            return Self {
                track_count: 0,
                total_duration_s: 0.0,
                avg_energy: 0.0,
                avg_transition_score: 0.0,
                distinct_genres: 0,
            };
        }

        let total_duration_s = playlist.iter().map(|t| t.duration_s).sum();
        let avg_energy = playlist.iter().map(|t| t.energy).sum::<f32>() / track_count as f32;

        let scorer = TransitionScorer::new();
        let transition_score_sum: f32 = playlist
            .windows(2)
            .map(|w| scorer.score(&w[0], &w[1]))
            .sum();
        let avg_transition_score = if track_count > 1 {
            transition_score_sum / (track_count - 1) as f32
        } else {
            1.0
        };

        let mut genres: HashSet<&str> = HashSet::new();
        for t in playlist {
            for g in &t.genres {
                genres.insert(g.as_str());
            }
        }

        Self {
            track_count,
            total_duration_s,
            avg_energy,
            avg_transition_score,
            distinct_genres: genres.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn rock() -> Vec<String> {
        vec!["rock".to_string()]
    }
    fn pop() -> Vec<String> {
        vec!["pop".to_string()]
    }
    fn jazz() -> Vec<String> {
        vec!["jazz".to_string()]
    }

    fn make_track(
        id: u64,
        genres: Vec<String>,
        energy: f32,
        tempo: f32,
        duration_s: f64,
    ) -> ContentTrack {
        ContentTrack::new(id, format!("Track {id}"))
            .with_genres(genres)
            .with_energy(energy)
            .with_tempo(tempo)
            .with_duration(duration_s)
            .with_popularity(0.5)
    }

    fn build_pool() -> PlaylistGenerator {
        let mut gen = PlaylistGenerator::new();
        gen.add_tracks(vec![
            make_track(1, rock(), 0.8, 130.0, 200.0),
            make_track(2, rock(), 0.7, 125.0, 210.0),
            make_track(3, pop(), 0.5, 120.0, 180.0),
            make_track(4, pop(), 0.4, 115.0, 195.0),
            make_track(5, jazz(), 0.3, 95.0, 250.0),
            make_track(6, jazz(), 0.2, 90.0, 240.0),
            make_track(7, rock(), 0.9, 140.0, 215.0),
            make_track(8, pop(), 0.6, 118.0, 185.0),
        ]);
        gen
    }

    #[test]
    fn test_generate_basic() {
        let gen = build_pool();
        let config = PlaylistConfig {
            max_tracks: 5,
            ..Default::default()
        };
        let playlist = gen
            .generate(&[1], &config)
            .expect("generate should succeed");
        assert!(!playlist.is_empty());
        assert!(playlist.len() <= 5);
    }

    #[test]
    fn test_no_duplicate_tracks() {
        let gen = build_pool();
        let config = PlaylistConfig {
            max_tracks: 8,
            ..Default::default()
        };
        let playlist = gen.generate(&[1], &config).expect("should succeed");
        let ids: HashSet<u64> = playlist.iter().map(|t| t.id).collect();
        assert_eq!(ids.len(), playlist.len(), "duplicate tracks in playlist");
    }

    #[test]
    fn test_duration_cap() {
        let gen = build_pool();
        let config = PlaylistConfig {
            max_duration_s: Some(600.0), // 10 minutes
            max_tracks: 100,
            ..Default::default()
        };
        let playlist = gen.generate(&[1], &config).expect("should succeed");
        let total: f64 = playlist.iter().map(|t| t.duration_s).sum();
        assert!(total <= 600.0 + f64::EPSILON);
    }

    #[test]
    fn test_empty_pool_returns_error() {
        let gen = PlaylistGenerator::new();
        let config = PlaylistConfig::default();
        let result = gen.generate(&[], &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_seed_first_in_playlist() {
        let gen = build_pool();
        let config = PlaylistConfig {
            max_tracks: 4,
            ..Default::default()
        };
        let playlist = gen.generate(&[5], &config).expect("should succeed");
        assert!(!playlist.is_empty());
        assert_eq!(playlist[0].id, 5, "seed should be first track");
    }

    #[test]
    fn test_energy_arc_rise_peak() {
        let arc = EnergyArc::RisePeak;
        let mid = arc.target_energy(5, 10);
        let start = arc.target_energy(0, 10);
        assert!(
            mid > start,
            "energy should be higher at midpoint than start"
        );
    }

    #[test]
    fn test_transition_scorer_same_genre() {
        let scorer = TransitionScorer::new();
        let a = make_track(1, rock(), 0.8, 130.0, 200.0);
        let b = make_track(2, rock(), 0.8, 130.0, 200.0);
        let c = make_track(3, jazz(), 0.2, 90.0, 200.0);
        let score_same = scorer.score(&a, &b);
        let score_diff = scorer.score(&a, &c);
        assert!(score_same > score_diff, "same genre should score higher");
    }

    #[test]
    fn test_playlist_stats_no_tracks() {
        let stats = PlaylistStats::compute(&[]);
        assert_eq!(stats.track_count, 0);
        assert_eq!(stats.total_duration_s, 0.0);
    }

    #[test]
    fn test_playlist_stats_multiple_tracks() {
        let tracks = vec![
            make_track(1, rock(), 0.8, 130.0, 200.0),
            make_track(2, rock(), 0.7, 125.0, 210.0),
            make_track(3, pop(), 0.5, 120.0, 180.0),
        ];
        let stats = PlaylistStats::compute(&tracks);
        assert_eq!(stats.track_count, 3);
        assert!((stats.total_duration_s - 590.0).abs() < 1e-6);
        assert!(stats.avg_energy > 0.0 && stats.avg_energy < 1.0);
        assert!(stats.avg_transition_score >= 0.0 && stats.avg_transition_score <= 1.0);
        assert_eq!(stats.distinct_genres, 2); // rock + pop
    }

    #[test]
    fn test_avoid_genre_repeats() {
        let gen = build_pool();
        let config = PlaylistConfig {
            max_tracks: 6,
            avoid_genre_repeats: true,
            ..Default::default()
        };
        let playlist = gen.generate(&[1], &config).expect("should succeed");
        for pair in playlist.windows(2) {
            let a_genres: HashSet<&str> = pair[0].genres.iter().map(String::as_str).collect();
            let b_genres: HashSet<&str> = pair[1].genres.iter().map(String::as_str).collect();
            assert!(
                a_genres.is_disjoint(&b_genres),
                "consecutive tracks should not share genres with avoid_genre_repeats=true"
            );
        }
    }
}
