//! Genre-configurable game highlight detector.
//!
//! Extends the basic highlight detection with per-genre thresholds so that
//! what constitutes a "highlight moment" differs meaningfully between, e.g.,
//! a fighting game (fast, noisy exchanges) and a strategy game (gradual
//! tension build-up).
//!
//! # Design
//!
//! - [`GameGenre`] enumerates common streaming game genres.
//! - [`GenreThresholds`] holds all tunable detection parameters for one genre.
//! - [`GenreHighlightDetector`] is the primary entry point.  It accepts a
//!   stream of [`HighlightSignal`] samples and emits [`GenreHighlightEvent`]s
//!   when the combined signal exceeds the genre-specific threshold.
//! - A sliding-window energy accumulator (`SignalWindow`) smooths noisy
//!   signals while respecting the genre's responsiveness requirements.
//! - All library code is `unwrap()`-free and `unsafe`-free.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// GameGenre
// ---------------------------------------------------------------------------

/// Broad game genre classifications used to select detection thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GameGenre {
    /// First-person or third-person shooters (CS:GO, Valorant, CoD).
    Shooter,
    /// Fighting games (Street Fighter, Mortal Kombat, Tekken).
    Fighting,
    /// Battle royale (Fortnite, Apex Legends, PUBG).
    BattleRoyale,
    /// Multiplayer online battle arena (LoL, Dota 2, SMITE).
    Moba,
    /// Real-time strategy (StarCraft, Age of Empires).
    Rts,
    /// Sports simulations (FIFA, NBA 2K, Rocket League).
    Sports,
    /// Horror / survival (Resident Evil, Phasmophobia).
    Horror,
    /// Speedrunning — any game with speedrun community.
    Speedrun,
    /// Casual / family / party games.
    Casual,
    /// User-defined genre with custom thresholds.
    Custom,
}

impl GameGenre {
    /// Human-readable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Shooter => "Shooter",
            Self::Fighting => "Fighting",
            Self::BattleRoyale => "Battle Royale",
            Self::Moba => "MOBA",
            Self::Rts => "RTS",
            Self::Sports => "Sports",
            Self::Horror => "Horror",
            Self::Speedrun => "Speedrun",
            Self::Casual => "Casual",
            Self::Custom => "Custom",
        }
    }

    /// Return the built-in default [`GenreThresholds`] for this genre.
    #[must_use]
    pub fn default_thresholds(self) -> GenreThresholds {
        match self {
            Self::Shooter => GenreThresholds {
                audio_rms_threshold: 0.55,
                chat_rate_multiplier: 2.5,
                kill_streak_min: 3,
                score_delta_threshold: 1,
                window_secs: 4,
                cooldown_secs: 8,
                min_combined_score: 0.60,
            },
            Self::Fighting => GenreThresholds {
                audio_rms_threshold: 0.65,
                chat_rate_multiplier: 3.0,
                kill_streak_min: 2,
                score_delta_threshold: 1,
                window_secs: 3,
                cooldown_secs: 5,
                min_combined_score: 0.55,
            },
            Self::BattleRoyale => GenreThresholds {
                audio_rms_threshold: 0.50,
                chat_rate_multiplier: 2.0,
                kill_streak_min: 2,
                score_delta_threshold: 1,
                window_secs: 6,
                cooldown_secs: 15,
                min_combined_score: 0.55,
            },
            Self::Moba => GenreThresholds {
                audio_rms_threshold: 0.45,
                chat_rate_multiplier: 2.0,
                kill_streak_min: 3,
                score_delta_threshold: 5,
                window_secs: 8,
                cooldown_secs: 20,
                min_combined_score: 0.50,
            },
            Self::Rts => GenreThresholds {
                audio_rms_threshold: 0.40,
                chat_rate_multiplier: 1.8,
                kill_streak_min: 5,
                score_delta_threshold: 10,
                window_secs: 10,
                cooldown_secs: 30,
                min_combined_score: 0.45,
            },
            Self::Sports => GenreThresholds {
                audio_rms_threshold: 0.60,
                chat_rate_multiplier: 2.5,
                kill_streak_min: 2,
                score_delta_threshold: 1,
                window_secs: 5,
                cooldown_secs: 10,
                min_combined_score: 0.55,
            },
            Self::Horror => GenreThresholds {
                audio_rms_threshold: 0.70,
                chat_rate_multiplier: 4.0,
                kill_streak_min: 1,
                score_delta_threshold: 1,
                window_secs: 3,
                cooldown_secs: 12,
                min_combined_score: 0.65,
            },
            Self::Speedrun => GenreThresholds {
                audio_rms_threshold: 0.45,
                chat_rate_multiplier: 2.0,
                kill_streak_min: 1,
                score_delta_threshold: 1,
                window_secs: 5,
                cooldown_secs: 10,
                min_combined_score: 0.50,
            },
            Self::Casual => GenreThresholds {
                audio_rms_threshold: 0.35,
                chat_rate_multiplier: 1.5,
                kill_streak_min: 1,
                score_delta_threshold: 1,
                window_secs: 6,
                cooldown_secs: 8,
                min_combined_score: 0.40,
            },
            Self::Custom => GenreThresholds::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// GenreThresholds
// ---------------------------------------------------------------------------

/// All tunable detection thresholds for a single game genre.
#[derive(Debug, Clone)]
pub struct GenreThresholds {
    /// Minimum normalised RMS energy (0.0–1.0) for the audio signal to
    /// contribute to excitement.
    pub audio_rms_threshold: f32,
    /// Chat message rate must be at least this multiple of the baseline rate
    /// to register as "hype".
    pub chat_rate_multiplier: f32,
    /// Minimum consecutive kill/elimination events to count as a kill-streak.
    pub kill_streak_min: u32,
    /// Minimum score delta in a single window to trigger a score-change event.
    pub score_delta_threshold: i64,
    /// Sliding window length in seconds over which signals are accumulated.
    pub window_secs: u64,
    /// Minimum seconds that must elapse between two emitted highlight events
    /// (prevents alert flooding).
    pub cooldown_secs: u64,
    /// Minimum combined excitement score (0.0–1.0) required to emit a
    /// highlight event.
    pub min_combined_score: f32,
}

impl Default for GenreThresholds {
    fn default() -> Self {
        Self {
            audio_rms_threshold: 0.50,
            chat_rate_multiplier: 2.0,
            kill_streak_min: 3,
            score_delta_threshold: 1,
            window_secs: 5,
            cooldown_secs: 10,
            min_combined_score: 0.50,
        }
    }
}

// ---------------------------------------------------------------------------
// HighlightSignal
// ---------------------------------------------------------------------------

/// An input signal sample fed into the detector.
#[derive(Debug, Clone)]
pub struct HighlightSignal {
    /// Elapsed stream time in milliseconds when this signal was sampled.
    pub timestamp_ms: u64,
    /// Normalised audio RMS energy in `[0.0, 1.0]`.
    pub audio_rms: f32,
    /// Current chat messages per second (raw rate, not multiplied).
    pub chat_rate: f32,
    /// Baseline chat rate (used for multiplier comparison).
    pub chat_baseline: f32,
    /// Score delta since the last sample (can be negative).
    pub score_delta: i64,
    /// Number of kills/eliminations this player achieved since last sample.
    pub kills: u32,
    /// Whether a significant game event occurred (boss kill, round win, etc.).
    pub special_event: bool,
}

// ---------------------------------------------------------------------------
// HighlightCategory
// ---------------------------------------------------------------------------

/// Category of a detected highlight moment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HighlightCategory {
    /// Kill streak detected.
    KillStreak {
        /// Number of kills in the streak.
        count: u32,
    },
    /// Significant score change.
    ScoreChange {
        /// Delta that triggered the event.
        delta: i64,
    },
    /// Chat hype spike.
    ChatHype,
    /// Audio excitement spike (crowd, commentary, SFX).
    AudioExcitement,
    /// Special in-game event (boss, clutch round, finish).
    SpecialEvent,
    /// Combined multi-signal excitement.
    MultiSignal,
}

// ---------------------------------------------------------------------------
// GenreHighlightEvent
// ---------------------------------------------------------------------------

/// A highlight event detected by the genre-aware detector.
#[derive(Debug, Clone)]
pub struct GenreHighlightEvent {
    /// When the highlight started (ms since stream start).
    pub timestamp_ms: u64,
    /// Combined excitement score in `[0.0, 1.0]`.
    pub score: f32,
    /// Primary category of this highlight.
    pub category: HighlightCategory,
    /// Genre that was active when the highlight was detected.
    pub genre: GameGenre,
}

// ---------------------------------------------------------------------------
// SignalWindow
// ---------------------------------------------------------------------------

/// A fixed-duration sliding window accumulator for signal samples.
struct SignalWindow {
    window_ms: u64,
    /// (timestamp_ms, audio_rms, chat_multiplied, kill_count, score_delta, special)
    samples: VecDeque<(u64, f32, f32, u32, i64, bool)>,
}

impl SignalWindow {
    fn new(window_secs: u64) -> Self {
        Self {
            window_ms: window_secs.saturating_mul(1000),
            samples: VecDeque::new(),
        }
    }

    fn push(
        &mut self,
        ts_ms: u64,
        audio: f32,
        chat_mul: f32,
        kills: u32,
        score: i64,
        special: bool,
    ) {
        self.prune(ts_ms);
        self.samples
            .push_back((ts_ms, audio, chat_mul, kills, score, special));
    }

    fn prune(&mut self, now_ms: u64) {
        let cutoff = now_ms.saturating_sub(self.window_ms);
        while self
            .samples
            .front()
            .map(|(t, ..)| *t < cutoff)
            .unwrap_or(false)
        {
            self.samples.pop_front();
        }
    }

    fn peak_audio(&self) -> f32 {
        self.samples
            .iter()
            .map(|(_, a, ..)| *a)
            .fold(0.0_f32, f32::max)
    }

    fn peak_chat_multiplier(&self) -> f32 {
        self.samples
            .iter()
            .map(|(_, _, c, ..)| *c)
            .fold(0.0_f32, f32::max)
    }

    fn total_kills(&self) -> u32 {
        self.samples.iter().map(|(_, _, _, k, ..)| k).sum()
    }

    fn total_score_delta(&self) -> i64 {
        self.samples.iter().map(|(_, _, _, _, s, _)| s).sum()
    }

    fn any_special(&self) -> bool {
        self.samples.iter().any(|(_, _, _, _, _, sp)| *sp)
    }

    fn len(&self) -> usize {
        self.samples.len()
    }
}

// ---------------------------------------------------------------------------
// GenreHighlightDetector
// ---------------------------------------------------------------------------

/// Genre-aware game highlight detector.
///
/// # Example
///
/// ```
/// use oximedia_gaming::genre_highlight::{
///     GameGenre, GenreHighlightDetector, HighlightSignal,
/// };
///
/// let mut detector = GenreHighlightDetector::new(GameGenre::Shooter);
///
/// let signal = HighlightSignal {
///     timestamp_ms: 5000,
///     audio_rms: 0.9,
///     chat_rate: 10.0,
///     chat_baseline: 2.0,
///     score_delta: 1,
///     kills: 4,
///     special_event: false,
/// };
///
/// let events = detector.push_signal(signal);
/// // May or may not produce a highlight depending on accumulated state.
/// let _ = events;
/// ```
pub struct GenreHighlightDetector {
    genre: GameGenre,
    thresholds: GenreThresholds,
    window: SignalWindow,
    /// Elapsed ms when the last highlight was emitted (for cooldown).
    last_emit_ms: Option<u64>,
    /// Total highlights emitted.
    highlight_count: u64,
}

impl GenreHighlightDetector {
    /// Create a detector for the given genre using its default thresholds.
    #[must_use]
    pub fn new(genre: GameGenre) -> Self {
        let thresholds = genre.default_thresholds();
        let window = SignalWindow::new(thresholds.window_secs);
        Self {
            genre,
            thresholds,
            window,
            last_emit_ms: None,
            highlight_count: 0,
        }
    }

    /// Create a detector with fully custom thresholds (still associated with a
    /// genre label for event tagging).
    #[must_use]
    pub fn with_thresholds(genre: GameGenre, thresholds: GenreThresholds) -> Self {
        let window = SignalWindow::new(thresholds.window_secs);
        Self {
            genre,
            thresholds,
            window,
            last_emit_ms: None,
            highlight_count: 0,
        }
    }

    /// Feed a signal sample and return any highlight events that should be
    /// emitted now.  At most one event is returned per call (cooldown enforced).
    pub fn push_signal(&mut self, signal: HighlightSignal) -> Vec<GenreHighlightEvent> {
        let ts = signal.timestamp_ms;

        // Compute chat multiplier (avoid division by zero).
        let chat_mul = if signal.chat_baseline > 0.0 {
            signal.chat_rate / signal.chat_baseline
        } else {
            0.0
        };

        self.window.push(
            ts,
            signal.audio_rms,
            chat_mul,
            signal.kills,
            signal.score_delta,
            signal.special_event,
        );

        // Enforce cooldown
        if let Some(last) = self.last_emit_ms {
            let cooldown_ms = self.thresholds.cooldown_secs.saturating_mul(1000);
            if ts.saturating_sub(last) < cooldown_ms {
                return vec![];
            }
        }

        // Evaluate whether thresholds are met and compute combined score.
        if let Some(event) = self.evaluate(ts) {
            self.last_emit_ms = Some(ts);
            self.highlight_count = self.highlight_count.saturating_add(1);
            vec![event]
        } else {
            vec![]
        }
    }

    /// Access the current genre.
    #[must_use]
    pub fn genre(&self) -> GameGenre {
        self.genre
    }

    /// Access the active thresholds.
    #[must_use]
    pub fn thresholds(&self) -> &GenreThresholds {
        &self.thresholds
    }

    /// Update the thresholds at runtime (e.g. after a genre switch).
    pub fn set_thresholds(&mut self, thresholds: GenreThresholds) {
        self.window = SignalWindow::new(thresholds.window_secs);
        self.thresholds = thresholds;
    }

    /// Switch to a different genre and reload default thresholds.
    pub fn set_genre(&mut self, genre: GameGenre) {
        self.genre = genre;
        self.set_thresholds(genre.default_thresholds());
    }

    /// Total highlight events emitted so far.
    #[must_use]
    pub fn highlight_count(&self) -> u64 {
        self.highlight_count
    }

    /// Reset detection state (e.g. on stream restart or game change).
    pub fn reset(&mut self) {
        self.window = SignalWindow::new(self.thresholds.window_secs);
        self.last_emit_ms = None;
        self.highlight_count = 0;
    }

    // ------------------------------------------------------------------
    // Private evaluation logic
    // ------------------------------------------------------------------

    fn evaluate(&self, ts: u64) -> Option<GenreHighlightEvent> {
        if self.window.len() == 0 {
            return None;
        }

        let t = &self.thresholds;

        // --- Individual signal scores (each in [0.0, 1.0]) ---

        // Audio: how much above threshold?
        let audio = self.window.peak_audio();
        let audio_score = if audio >= t.audio_rms_threshold {
            ((audio - t.audio_rms_threshold) / (1.0 - t.audio_rms_threshold).max(1e-6)).min(1.0)
        } else {
            0.0
        };

        // Chat hype: how many multiples above the multiplier threshold?
        let chat_mul = self.window.peak_chat_multiplier();
        let chat_score = if chat_mul >= t.chat_rate_multiplier {
            ((chat_mul - t.chat_rate_multiplier) / t.chat_rate_multiplier).min(1.0)
        } else {
            0.0
        };

        // Kill streak
        let kills = self.window.total_kills();
        let kill_score = if kills >= t.kill_streak_min {
            ((kills - t.kill_streak_min + 1) as f32 / (t.kill_streak_min as f32).max(1.0)).min(1.0)
        } else {
            0.0
        };

        // Score change
        let score_delta = self.window.total_score_delta().abs();
        let score_change_score = if score_delta >= t.score_delta_threshold.abs() {
            (score_delta as f32 / (t.score_delta_threshold.abs() as f32).max(1.0)).min(1.0)
        } else {
            0.0
        };

        // Special event: binary
        let special_score: f32 = if self.window.any_special() { 1.0 } else { 0.0 };

        // Combined weighted score
        let combined = audio_score * 0.30
            + chat_score * 0.25
            + kill_score * 0.25
            + score_change_score * 0.10
            + special_score * 0.10;

        if combined < t.min_combined_score {
            return None;
        }

        // Determine primary category (highest individual score)
        let scores = [
            (audio_score, HighlightCategory::AudioExcitement),
            (chat_score, HighlightCategory::ChatHype),
            (kill_score, HighlightCategory::KillStreak { count: kills }),
            (
                score_change_score,
                HighlightCategory::ScoreChange {
                    delta: self.window.total_score_delta(),
                },
            ),
            (special_score, HighlightCategory::SpecialEvent),
        ];

        // Count how many signals are active (score > 0)
        let active: u32 = scores.iter().filter(|(s, _)| *s > 0.0).count() as u32;

        let category = if active >= 3 {
            HighlightCategory::MultiSignal
        } else {
            scores
                .into_iter()
                .max_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(_, c)| c)
                .unwrap_or(HighlightCategory::MultiSignal)
        };

        Some(GenreHighlightEvent {
            timestamp_ms: ts,
            score: combined,
            category,
            genre: self.genre,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn loud_signal(ts_ms: u64) -> HighlightSignal {
        HighlightSignal {
            timestamp_ms: ts_ms,
            audio_rms: 0.95,
            chat_rate: 20.0,
            chat_baseline: 2.0,
            score_delta: 5,
            kills: 5,
            special_event: true,
        }
    }

    fn quiet_signal(ts_ms: u64) -> HighlightSignal {
        HighlightSignal {
            timestamp_ms: ts_ms,
            audio_rms: 0.1,
            chat_rate: 1.0,
            chat_baseline: 2.0,
            score_delta: 0,
            kills: 0,
            special_event: false,
        }
    }

    #[test]
    fn test_genre_names_non_empty() {
        let genres = [
            GameGenre::Shooter,
            GameGenre::Fighting,
            GameGenre::BattleRoyale,
            GameGenre::Moba,
            GameGenre::Rts,
            GameGenre::Sports,
            GameGenre::Horror,
            GameGenre::Speedrun,
            GameGenre::Casual,
            GameGenre::Custom,
        ];
        for g in genres {
            assert!(
                !g.name().is_empty(),
                "genre name should not be empty: {g:?}"
            );
        }
    }

    #[test]
    fn test_loud_signal_triggers_highlight_shooter() {
        let mut det = GenreHighlightDetector::new(GameGenre::Shooter);
        let events = det.push_signal(loud_signal(5000));
        assert!(
            !events.is_empty(),
            "should detect highlight from loud signal"
        );
        assert!(events[0].score >= det.thresholds().min_combined_score);
    }

    #[test]
    fn test_quiet_signal_no_highlight() {
        let mut det = GenreHighlightDetector::new(GameGenre::Shooter);
        let events = det.push_signal(quiet_signal(1000));
        assert!(
            events.is_empty(),
            "quiet signal should not produce a highlight"
        );
    }

    #[test]
    fn test_cooldown_prevents_flood() {
        let mut det = GenreHighlightDetector::new(GameGenre::Shooter);
        // First loud signal — should trigger
        let first = det.push_signal(loud_signal(5000));
        assert!(!first.is_empty());
        // Immediately after — should be suppressed by cooldown
        let second = det.push_signal(loud_signal(5100));
        assert!(second.is_empty(), "should be suppressed by cooldown");
    }

    #[test]
    fn test_cooldown_expires() {
        let mut det = GenreHighlightDetector::new(GameGenre::Shooter);
        let _ = det.push_signal(loud_signal(0));
        let cooldown_ms = det.thresholds().cooldown_secs * 1000;
        // After cooldown passes, a new loud signal should trigger
        let late = det.push_signal(loud_signal(cooldown_ms + 1));
        assert!(!late.is_empty(), "highlight should fire after cooldown");
    }

    #[test]
    fn test_genre_switch_resets_thresholds() {
        let mut det = GenreHighlightDetector::new(GameGenre::Shooter);
        det.set_genre(GameGenre::Rts);
        assert_eq!(det.genre(), GameGenre::Rts);
        // RTS has a longer window — just verify thresholds were updated
        assert!(det.thresholds().window_secs >= 8);
    }

    #[test]
    fn test_custom_thresholds_very_low_trigger() {
        let low = GenreThresholds {
            audio_rms_threshold: 0.01,
            chat_rate_multiplier: 0.01,
            kill_streak_min: 1,
            score_delta_threshold: 0,
            window_secs: 5,
            cooldown_secs: 0,
            min_combined_score: 0.01,
        };
        let mut det = GenreHighlightDetector::with_thresholds(GameGenre::Custom, low);
        let events = det.push_signal(quiet_signal(1000));
        // Even a quiet signal should trigger with very low thresholds
        assert!(!events.is_empty());
    }

    #[test]
    fn test_highlight_count_increments() {
        let mut det = GenreHighlightDetector::new(GameGenre::Shooter);
        let cooldown_ms = det.thresholds().cooldown_secs * 1000 + 1;
        det.push_signal(loud_signal(0));
        det.push_signal(loud_signal(cooldown_ms));
        assert_eq!(det.highlight_count(), 2);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut det = GenreHighlightDetector::new(GameGenre::Shooter);
        det.push_signal(loud_signal(1000));
        det.reset();
        assert_eq!(det.highlight_count(), 0);
        assert!(det.last_emit_ms.is_none());
    }

    #[test]
    fn test_horror_genre_higher_audio_threshold() {
        let horror_thresh = GameGenre::Horror.default_thresholds().audio_rms_threshold;
        let casual_thresh = GameGenre::Casual.default_thresholds().audio_rms_threshold;
        assert!(
            horror_thresh > casual_thresh,
            "horror should require louder audio than casual"
        );
    }

    #[test]
    fn test_multi_signal_category_when_many_active() {
        let mut det = GenreHighlightDetector::new(GameGenre::Shooter);
        // Loud signal activates audio, chat, kills, score, and special all at once
        let events = det.push_signal(loud_signal(5000));
        if !events.is_empty() {
            // Should be classified as MultiSignal since all signals are active
            assert!(matches!(events[0].category, HighlightCategory::MultiSignal));
        }
    }

    #[test]
    fn test_kill_streak_category_kills_only() {
        let t = GenreThresholds {
            audio_rms_threshold: 0.99,   // unreachable
            chat_rate_multiplier: 100.0, // unreachable
            kill_streak_min: 2,
            score_delta_threshold: 999, // unreachable
            window_secs: 5,
            cooldown_secs: 0,
            min_combined_score: 0.01,
        };
        let mut det = GenreHighlightDetector::with_thresholds(GameGenre::Custom, t);
        let signal = HighlightSignal {
            timestamp_ms: 1000,
            audio_rms: 0.0,
            chat_rate: 0.0,
            chat_baseline: 1.0,
            score_delta: 0,
            kills: 3,
            special_event: false,
        };
        let events = det.push_signal(signal);
        assert!(!events.is_empty());
        assert!(
            matches!(
                events[0].category,
                HighlightCategory::KillStreak { count: 3 }
            ),
            "category should be KillStreak, got {:?}",
            events[0].category
        );
    }
}
