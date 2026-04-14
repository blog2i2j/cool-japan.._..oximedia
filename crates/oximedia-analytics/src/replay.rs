//! Session replay reconstruction from `PlaybackEvent` sequences.
//!
//! The replay module transforms a raw `Vec<PlaybackEvent>` into a structured,
//! ordered timeline of `ReplayFrame` entries that describe **what was happening
//! at each millisecond boundary** during a viewing session.
//!
//! ## Use-cases
//!
//! * **Debugging** — reproduce exactly which content positions a viewer visited
//!   and in which order, including seeks, stalls, and quality switches.
//! * **QoE analysis** — measure time-to-first-frame, stall frequency, and
//!   quality-level distributions across a reconstructed timeline.
//! * **Session annotation** — attach human-readable labels to each event for
//!   side-by-side comparison with server-side logs.
//!
//! ## Algorithm
//!
//! 1. Walk the `PlaybackEvent` list in order.
//! 2. Maintain a virtual "player state machine" (`PlayerState`).
//! 3. At each event transition, emit a `ReplayFrame` capturing the state
//!    immediately *before* and *after* the event.
//! 4. Optionally interpolate intermediate frames at a configurable resolution
//!    so downstream code can render a smooth timeline.

use crate::error::AnalyticsError;
use crate::session::{PlaybackEvent, ViewerSession};

// ─── Player state ────────────────────────────────────────────────────────────

/// The high-level playback state of the player at a point in time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlayerState {
    /// Player has not yet started (initial state).
    Idle,
    /// Content is actively playing.
    Playing,
    /// Playback is paused.
    Paused,
    /// Player is buffering (stalled waiting for data).
    Buffering,
    /// Playback has ended (reached the end or was explicitly closed).
    Ended,
}

impl std::fmt::Display for PlayerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::Playing => write!(f, "Playing"),
            Self::Paused => write!(f, "Paused"),
            Self::Buffering => write!(f, "Buffering"),
            Self::Ended => write!(f, "Ended"),
        }
    }
}

// ─── ReplayFrame ─────────────────────────────────────────────────────────────

/// A single frame in the reconstructed session replay timeline.
///
/// Each `ReplayFrame` represents a **discrete event transition** — the moment
/// a `PlaybackEvent` changes the player state.  Optional interpolated frames
/// (produced by [`ReplayReconstructor::interpolate`]) have `interpolated = true`
/// and carry no `event_kind`.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayFrame {
    /// Wall-clock offset from session start (milliseconds).
    pub wall_ms: i64,
    /// Content position at this frame (milliseconds into the content).
    pub content_pos_ms: u64,
    /// Player state at this frame.
    pub state: PlayerState,
    /// Current video quality height in pixels (e.g. 1080, 720, 480).
    /// `None` until the first `QualityChange` event.
    pub quality_height: Option<u32>,
    /// Current bitrate reported by the last `QualityChange` event (bps).
    pub bitrate_bps: Option<u32>,
    /// Human-readable label of the triggering event (e.g. `"Play"`, `"Seek"`).
    /// Empty string for interpolated frames.
    pub event_kind: String,
    /// `true` if this frame was synthetically inserted by interpolation.
    pub interpolated: bool,
    /// Index of the source event in the original `PlaybackEvent` list (0-based).
    /// `None` for interpolated frames.
    pub source_event_index: Option<usize>,
}

impl ReplayFrame {
    fn event(
        wall_ms: i64,
        content_pos_ms: u64,
        state: PlayerState,
        quality_height: Option<u32>,
        bitrate_bps: Option<u32>,
        event_kind: &str,
        source_event_index: usize,
    ) -> Self {
        Self {
            wall_ms,
            content_pos_ms,
            state,
            quality_height,
            bitrate_bps,
            event_kind: event_kind.to_string(),
            interpolated: false,
            source_event_index: Some(source_event_index),
        }
    }

    fn interpolated_frame(
        wall_ms: i64,
        content_pos_ms: u64,
        state: PlayerState,
        quality_height: Option<u32>,
        bitrate_bps: Option<u32>,
    ) -> Self {
        Self {
            wall_ms,
            content_pos_ms,
            state,
            quality_height,
            bitrate_bps,
            event_kind: String::new(),
            interpolated: true,
            source_event_index: None,
        }
    }
}

// ─── Session summary ─────────────────────────────────────────────────────────

/// High-level quality-of-experience summary derived from a reconstructed replay.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplaySummary {
    /// Total number of `ReplayFrame` entries in the timeline.
    pub frame_count: usize,
    /// Number of seek events.
    pub seek_count: u32,
    /// Number of buffering episodes.
    pub stall_count: u32,
    /// Total stall duration in milliseconds (sum of `BufferEnd.duration_ms`).
    pub total_stall_ms: u64,
    /// Number of quality level changes.
    pub quality_change_count: u32,
    /// Minimum quality height seen, if any quality events were present.
    pub min_quality_height: Option<u32>,
    /// Maximum quality height seen, if any quality events were present.
    pub max_quality_height: Option<u32>,
    /// Final player state at the end of the timeline.
    pub final_state: PlayerState,
    /// Furthest content position reached (ms).
    pub max_content_pos_ms: u64,
}

// ─── ReplayReconstructor ────────────────────────────────────────────────────

/// Configuration for the replay reconstruction.
#[derive(Debug, Clone)]
pub struct ReplayConfig {
    /// If `Some(step_ms)`, interpolated frames are inserted every `step_ms`
    /// while in the `Playing` state to produce a smooth timeline.
    /// Set to `None` (default) to emit event frames only.
    pub interpolation_step_ms: Option<u64>,
    /// Wall-clock rate of advance for interpolation: how many milliseconds of
    /// content advance per millisecond of wall-clock time when playing normally.
    /// Default is `1.0` (real-time playback).
    pub playback_rate: f64,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            interpolation_step_ms: None,
            playback_rate: 1.0,
        }
    }
}

/// Reconstructs a structured replay timeline from a `ViewerSession`.
///
/// # Example
///
/// ```rust
/// use oximedia_analytics::replay::{ReplayReconstructor, ReplayConfig};
/// use oximedia_analytics::session::{ViewerSession, PlaybackEvent};
///
/// let session = ViewerSession {
///     session_id: "s1".to_string(),
///     user_id: None,
///     content_id: "vid_001".to_string(),
///     started_at_ms: 0,
///     events: vec![
///         PlaybackEvent::Play { timestamp_ms: 0 },
///         PlaybackEvent::Pause { timestamp_ms: 5000, position_ms: 5000 },
///         PlaybackEvent::End { position_ms: 5000, watch_duration_ms: 5000 },
///     ],
/// };
///
/// let rec = ReplayReconstructor::new(ReplayConfig::default());
/// let timeline = rec.reconstruct(&session).unwrap();
/// assert!(!timeline.is_empty());
/// ```
pub struct ReplayReconstructor {
    config: ReplayConfig,
}

impl ReplayReconstructor {
    /// Create a new reconstructor with the given configuration.
    pub fn new(config: ReplayConfig) -> Self {
        Self { config }
    }

    /// Reconstruct a replay timeline from a `ViewerSession`.
    ///
    /// Returns an ordered `Vec<ReplayFrame>` sorted by `wall_ms`.
    ///
    /// Returns an error if the session has no events.
    pub fn reconstruct(&self, session: &ViewerSession) -> Result<Vec<ReplayFrame>, AnalyticsError> {
        if session.events.is_empty() {
            return Err(AnalyticsError::InsufficientData(
                "session has no playback events".to_string(),
            ));
        }

        let mut frames: Vec<ReplayFrame> = Vec::with_capacity(session.events.len() * 2);

        // Player state machine.
        let mut state = PlayerState::Idle;
        let mut content_pos_ms: u64 = 0;
        let mut quality_height: Option<u32> = None;
        let mut bitrate_bps: Option<u32> = None;
        // Wall-clock offset from session start (we advance it heuristically).
        let mut wall_ms: i64 = 0;
        let mut play_start_wall_ms: i64 = 0;
        let mut play_start_content_ms: u64 = 0;

        let playback_rate = self.config.playback_rate.max(0.001);

        for (idx, event) in session.events.iter().enumerate() {
            match event {
                PlaybackEvent::Play { timestamp_ms } => {
                    // If timestamp is available from the event, use it.
                    if *timestamp_ms >= 0 {
                        wall_ms = *timestamp_ms - session.started_at_ms;
                    }
                    play_start_wall_ms = wall_ms;
                    play_start_content_ms = content_pos_ms;

                    // Optionally insert interpolated frames up to this point.
                    if state == PlayerState::Playing {
                        // Re-play event while already playing — update state.
                    }
                    state = PlayerState::Playing;

                    frames.push(ReplayFrame::event(
                        wall_ms,
                        content_pos_ms,
                        state,
                        quality_height,
                        bitrate_bps,
                        "Play",
                        idx,
                    ));
                }

                PlaybackEvent::Pause { timestamp_ms, position_ms } => {
                    // Advance wall clock based on content position change.
                    if *timestamp_ms >= 0 {
                        wall_ms = *timestamp_ms - session.started_at_ms;
                    } else if state == PlayerState::Playing {
                        let content_delta = position_ms.saturating_sub(play_start_content_ms);
                        wall_ms = play_start_wall_ms
                            + (content_delta as f64 / playback_rate) as i64;
                    }

                    if state == PlayerState::Playing {
                        // Insert interpolation frames before the pause.
                        if let Some(step) = self.config.interpolation_step_ms {
                            frames.extend(make_interpolated_frames(
                                play_start_wall_ms,
                                play_start_content_ms,
                                wall_ms,
                                *position_ms,
                                PlayerState::Playing,
                                quality_height,
                                bitrate_bps,
                                step,
                            ));
                        }
                    }

                    content_pos_ms = *position_ms;
                    state = PlayerState::Paused;

                    frames.push(ReplayFrame::event(
                        wall_ms,
                        content_pos_ms,
                        state,
                        quality_height,
                        bitrate_bps,
                        "Pause",
                        idx,
                    ));
                }

                PlaybackEvent::Seek { from_ms, to_ms } => {
                    if state == PlayerState::Playing {
                        let content_delta = from_ms.saturating_sub(play_start_content_ms);
                        wall_ms = play_start_wall_ms
                            + (content_delta as f64 / playback_rate) as i64;

                        if let Some(step) = self.config.interpolation_step_ms {
                            frames.extend(make_interpolated_frames(
                                play_start_wall_ms,
                                play_start_content_ms,
                                wall_ms,
                                *from_ms,
                                PlayerState::Playing,
                                quality_height,
                                bitrate_bps,
                                step,
                            ));
                        }
                    }

                    content_pos_ms = *to_ms;
                    // After a seek while playing, update play_start references.
                    if state == PlayerState::Playing {
                        play_start_wall_ms = wall_ms;
                        play_start_content_ms = *to_ms;
                    }

                    frames.push(ReplayFrame::event(
                        wall_ms,
                        content_pos_ms,
                        state,
                        quality_height,
                        bitrate_bps,
                        "Seek",
                        idx,
                    ));
                }

                PlaybackEvent::BufferStart { position_ms } => {
                    if state == PlayerState::Playing {
                        let content_delta = position_ms.saturating_sub(play_start_content_ms);
                        wall_ms = play_start_wall_ms
                            + (content_delta as f64 / playback_rate) as i64;
                    }

                    content_pos_ms = *position_ms;
                    state = PlayerState::Buffering;

                    frames.push(ReplayFrame::event(
                        wall_ms,
                        content_pos_ms,
                        state,
                        quality_height,
                        bitrate_bps,
                        "BufferStart",
                        idx,
                    ));
                }

                PlaybackEvent::BufferEnd { position_ms, duration_ms } => {
                    // Advance wall clock by stall duration.
                    wall_ms += i64::from(*duration_ms);
                    content_pos_ms = *position_ms;
                    state = PlayerState::Playing;
                    play_start_wall_ms = wall_ms;
                    play_start_content_ms = *position_ms;

                    frames.push(ReplayFrame::event(
                        wall_ms,
                        content_pos_ms,
                        state,
                        quality_height,
                        bitrate_bps,
                        "BufferEnd",
                        idx,
                    ));
                }

                PlaybackEvent::QualityChange { from_height: _, to_height, bitrate } => {
                    if state == PlayerState::Playing {
                        // We can't know exact wall time without a timestamp in
                        // QualityChange, so we don't advance the clock here.
                    }

                    // Track min/max quality heights.
                    quality_height = Some(*to_height);
                    bitrate_bps = Some(*bitrate);

                    frames.push(ReplayFrame::event(
                        wall_ms,
                        content_pos_ms,
                        state,
                        quality_height,
                        bitrate_bps,
                        "QualityChange",
                        idx,
                    ));
                }

                PlaybackEvent::End { position_ms, watch_duration_ms } => {
                    if state == PlayerState::Playing {
                        // Use watch_duration_ms to compute wall-clock end.
                        let content_delta = position_ms.saturating_sub(play_start_content_ms);
                        wall_ms = play_start_wall_ms
                            + (content_delta as f64 / playback_rate) as i64;

                        if let Some(step) = self.config.interpolation_step_ms {
                            frames.extend(make_interpolated_frames(
                                play_start_wall_ms,
                                play_start_content_ms,
                                wall_ms,
                                *position_ms,
                                PlayerState::Playing,
                                quality_height,
                                bitrate_bps,
                                step,
                            ));
                        }
                        // Also use watch_duration_ms as a sanity override.
                        wall_ms = wall_ms.max(*watch_duration_ms as i64);
                    }

                    content_pos_ms = *position_ms;
                    state = PlayerState::Ended;

                    frames.push(ReplayFrame::event(
                        wall_ms,
                        content_pos_ms,
                        state,
                        quality_height,
                        bitrate_bps,
                        "End",
                        idx,
                    ));
                }
            }
        }

        // Sort by wall_ms to handle any out-of-order events.
        frames.sort_by_key(|f| f.wall_ms);

        Ok(frames)
    }

    /// Compute a high-level quality-of-experience summary from a timeline.
    pub fn summarise(frames: &[ReplayFrame], session: &ViewerSession) -> ReplaySummary {
        let mut seek_count: u32 = 0;
        let mut stall_count: u32 = 0;
        let mut total_stall_ms: u64 = 0;
        let mut quality_change_count: u32 = 0;
        let mut min_quality_height: Option<u32> = None;
        let mut max_quality_height: Option<u32> = None;
        let mut max_content_pos_ms: u64 = 0;

        // Count events from original session (more reliable than derived frames).
        for event in &session.events {
            match event {
                PlaybackEvent::Seek { .. } => seek_count += 1,
                PlaybackEvent::BufferStart { .. } => stall_count += 1,
                PlaybackEvent::BufferEnd { duration_ms, .. } => {
                    total_stall_ms += u64::from(*duration_ms);
                }
                PlaybackEvent::QualityChange { to_height, .. } => {
                    quality_change_count += 1;
                    min_quality_height = Some(
                        min_quality_height.map_or(*to_height, |m: u32| m.min(*to_height)),
                    );
                    max_quality_height = Some(
                        max_quality_height.map_or(*to_height, |m: u32| m.max(*to_height)),
                    );
                }
                _ => {}
            }
        }

        for f in frames {
            if f.content_pos_ms > max_content_pos_ms {
                max_content_pos_ms = f.content_pos_ms;
            }
        }

        let final_state = frames.last().map(|f| f.state).unwrap_or(PlayerState::Idle);

        ReplaySummary {
            frame_count: frames.len(),
            seek_count,
            stall_count,
            total_stall_ms,
            quality_change_count,
            min_quality_height,
            max_quality_height,
            final_state,
            max_content_pos_ms,
        }
    }
}

// ─── Interpolation helper ─────────────────────────────────────────────────────

/// Generate interpolated frames between two wall-clock/content-position points.
///
/// Frames are generated at `step_ms` wall-clock intervals between `wall_start`
/// (exclusive) and `wall_end` (exclusive).  Content position is linearly
/// interpolated between `content_start` and `content_end`.
fn make_interpolated_frames(
    wall_start: i64,
    content_start: u64,
    wall_end: i64,
    content_end: u64,
    state: PlayerState,
    quality_height: Option<u32>,
    bitrate_bps: Option<u32>,
    step_ms: u64,
) -> Vec<ReplayFrame> {
    if step_ms == 0 || wall_end <= wall_start {
        return Vec::new();
    }

    let wall_duration = (wall_end - wall_start) as f64;
    let content_duration = content_end as f64 - content_start as f64;

    let step_i64 = step_ms as i64;
    let mut frames = Vec::new();
    let mut t = wall_start + step_i64;

    while t < wall_end {
        let progress = (t - wall_start) as f64 / wall_duration;
        let content_pos = (content_start as f64 + progress * content_duration).max(0.0) as u64;

        frames.push(ReplayFrame::interpolated_frame(
            t,
            content_pos,
            state,
            quality_height,
            bitrate_bps,
        ));
        t += step_i64;
    }

    frames
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{PlaybackEvent, ViewerSession};

    fn make_session(id: &str, events: Vec<PlaybackEvent>) -> ViewerSession {
        ViewerSession {
            session_id: id.to_string(),
            user_id: None,
            content_id: "vid_001".to_string(),
            started_at_ms: 0,
            events,
        }
    }

    // ── basic reconstruction ─────────────────────────────────────────────────

    #[test]
    fn empty_session_returns_error() {
        let session = make_session("s1", vec![]);
        let rec = ReplayReconstructor::new(ReplayConfig::default());
        assert!(rec.reconstruct(&session).is_err());
    }

    #[test]
    fn single_play_event_produces_playing_frame() {
        let session = make_session(
            "s1",
            vec![PlaybackEvent::Play { timestamp_ms: 0 }],
        );
        let rec = ReplayReconstructor::new(ReplayConfig::default());
        let frames = rec.reconstruct(&session).expect("should succeed");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].state, PlayerState::Playing);
        assert_eq!(frames[0].event_kind, "Play");
    }

    #[test]
    fn play_end_sequence_ends_in_ended_state() {
        let session = make_session(
            "s1",
            vec![
                PlaybackEvent::Play { timestamp_ms: 0 },
                PlaybackEvent::End {
                    position_ms: 30_000,
                    watch_duration_ms: 30_000,
                },
            ],
        );
        let rec = ReplayReconstructor::new(ReplayConfig::default());
        let frames = rec.reconstruct(&session).expect("should succeed");
        let last = frames.last().expect("at least one frame");
        assert_eq!(last.state, PlayerState::Ended);
        assert_eq!(last.content_pos_ms, 30_000);
    }

    #[test]
    fn pause_event_transitions_to_paused() {
        let session = make_session(
            "s1",
            vec![
                PlaybackEvent::Play { timestamp_ms: 0 },
                PlaybackEvent::Pause {
                    timestamp_ms: 5_000,
                    position_ms: 5_000,
                },
            ],
        );
        let rec = ReplayReconstructor::new(ReplayConfig::default());
        let frames = rec.reconstruct(&session).expect("should succeed");
        let paused = frames.iter().find(|f| f.state == PlayerState::Paused);
        assert!(paused.is_some(), "expected a Paused frame");
        assert_eq!(paused.unwrap().content_pos_ms, 5_000);
    }

    #[test]
    fn buffer_start_transitions_to_buffering() {
        let session = make_session(
            "s1",
            vec![
                PlaybackEvent::Play { timestamp_ms: 0 },
                PlaybackEvent::BufferStart { position_ms: 10_000 },
                PlaybackEvent::BufferEnd {
                    position_ms: 10_000,
                    duration_ms: 500,
                },
            ],
        );
        let rec = ReplayReconstructor::new(ReplayConfig::default());
        let frames = rec.reconstruct(&session).expect("should succeed");
        let buffering = frames.iter().any(|f| f.state == PlayerState::Buffering);
        assert!(buffering, "expected a Buffering frame");
    }

    #[test]
    fn seek_event_updates_content_position() {
        let session = make_session(
            "s1",
            vec![
                PlaybackEvent::Play { timestamp_ms: 0 },
                PlaybackEvent::Seek {
                    from_ms: 10_000,
                    to_ms: 60_000,
                },
                PlaybackEvent::End {
                    position_ms: 90_000,
                    watch_duration_ms: 30_000,
                },
            ],
        );
        let rec = ReplayReconstructor::new(ReplayConfig::default());
        let frames = rec.reconstruct(&session).expect("should succeed");
        let seek_frame = frames.iter().find(|f| f.event_kind == "Seek");
        assert!(seek_frame.is_some());
        assert_eq!(seek_frame.unwrap().content_pos_ms, 60_000);
    }

    #[test]
    fn quality_change_tracked_in_frame() {
        let session = make_session(
            "s1",
            vec![
                PlaybackEvent::Play { timestamp_ms: 0 },
                PlaybackEvent::QualityChange {
                    from_height: 480,
                    to_height: 1080,
                    bitrate: 5_000_000,
                },
                PlaybackEvent::End {
                    position_ms: 20_000,
                    watch_duration_ms: 20_000,
                },
            ],
        );
        let rec = ReplayReconstructor::new(ReplayConfig::default());
        let frames = rec.reconstruct(&session).expect("should succeed");
        let qc = frames.iter().find(|f| f.event_kind == "QualityChange");
        assert!(qc.is_some());
        assert_eq!(qc.unwrap().quality_height, Some(1080));
        assert_eq!(qc.unwrap().bitrate_bps, Some(5_000_000));
    }

    // ── interpolation ────────────────────────────────────────────────────────

    #[test]
    fn interpolation_inserts_intermediate_frames() {
        let session = make_session(
            "s1",
            vec![
                PlaybackEvent::Play { timestamp_ms: 0 },
                PlaybackEvent::End {
                    position_ms: 10_000,
                    watch_duration_ms: 10_000,
                },
            ],
        );
        let config = ReplayConfig {
            interpolation_step_ms: Some(1_000),
            playback_rate: 1.0,
        };
        let rec = ReplayReconstructor::new(config);
        let frames = rec.reconstruct(&session).expect("should succeed");
        // Should have Play frame, ~9 interpolated frames (1s..9s), and End frame.
        let interp_count = frames.iter().filter(|f| f.interpolated).count();
        assert!(interp_count >= 1, "expected interpolated frames, got 0");
    }

    #[test]
    fn frames_sorted_by_wall_ms() {
        let session = make_session(
            "s1",
            vec![
                PlaybackEvent::Play { timestamp_ms: 0 },
                PlaybackEvent::Pause {
                    timestamp_ms: 5_000,
                    position_ms: 5_000,
                },
                PlaybackEvent::Play { timestamp_ms: 7_000 },
                PlaybackEvent::End {
                    position_ms: 20_000,
                    watch_duration_ms: 18_000,
                },
            ],
        );
        let rec = ReplayReconstructor::new(ReplayConfig::default());
        let frames = rec.reconstruct(&session).expect("should succeed");
        for w in frames.windows(2) {
            assert!(
                w[0].wall_ms <= w[1].wall_ms,
                "frames out of order: {} > {}",
                w[0].wall_ms,
                w[1].wall_ms
            );
        }
    }

    // ── summary ──────────────────────────────────────────────────────────────

    #[test]
    fn summary_counts_stalls_and_seeks() {
        let session = make_session(
            "s1",
            vec![
                PlaybackEvent::Play { timestamp_ms: 0 },
                PlaybackEvent::BufferStart { position_ms: 5_000 },
                PlaybackEvent::BufferEnd {
                    position_ms: 5_000,
                    duration_ms: 800,
                },
                PlaybackEvent::Seek {
                    from_ms: 10_000,
                    to_ms: 20_000,
                },
                PlaybackEvent::End {
                    position_ms: 30_000,
                    watch_duration_ms: 25_000,
                },
            ],
        );
        let rec = ReplayReconstructor::new(ReplayConfig::default());
        let frames = rec.reconstruct(&session).expect("should succeed");
        let summary = ReplayReconstructor::summarise(&frames, &session);
        assert_eq!(summary.stall_count, 1);
        assert_eq!(summary.total_stall_ms, 800);
        assert_eq!(summary.seek_count, 1);
        assert_eq!(summary.final_state, PlayerState::Ended);
    }

    #[test]
    fn summary_quality_min_max() {
        let session = make_session(
            "s1",
            vec![
                PlaybackEvent::Play { timestamp_ms: 0 },
                PlaybackEvent::QualityChange {
                    from_height: 480,
                    to_height: 720,
                    bitrate: 2_000_000,
                },
                PlaybackEvent::QualityChange {
                    from_height: 720,
                    to_height: 1080,
                    bitrate: 5_000_000,
                },
                PlaybackEvent::End {
                    position_ms: 60_000,
                    watch_duration_ms: 60_000,
                },
            ],
        );
        let rec = ReplayReconstructor::new(ReplayConfig::default());
        let frames = rec.reconstruct(&session).expect("should succeed");
        let summary = ReplayReconstructor::summarise(&frames, &session);
        assert_eq!(summary.quality_change_count, 2);
        assert_eq!(summary.min_quality_height, Some(720));
        assert_eq!(summary.max_quality_height, Some(1080));
    }
}
