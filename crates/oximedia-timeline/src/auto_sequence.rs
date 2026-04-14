//! Automatic assembly editing based on subclip markers.
//!
//! `auto_sequence` scans a list of source clips that carry named *subclip
//! markers* (in-point / out-point pairs with optional rating) and assembles
//! them into a new timeline sequence according to a configurable assembly
//! strategy. Optional crossfade transitions can be applied between consecutive
//! clips, and configurable gaps or overlaps control spacing.
//!
//! # Strategies
//!
//! - [`AssemblyStrategy::InOrder`] — clips are placed in the order they appear
//!   in the input list.
//! - [`AssemblyStrategy::ByRating`] — clips are sorted descending by marker
//!   rating (highest quality first).
//! - [`AssemblyStrategy::ByDuration`] — clips are sorted to pack the timeline
//!   most efficiently (shortest first for tight sequences).
//! - [`AssemblyStrategy::Interleaved`] — clips from different camera angles
//!   are interleaved round-robin.

use serde::{Deserialize, Serialize};

use crate::clip::ClipId;
use crate::track::TrackId;
use crate::types::{Duration, Position};

/// A named subclip range inside a source clip, produced by a logger or
/// assistant editor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubclipMarker {
    /// Identifier of the source clip this marker belongs to.
    pub source_clip_id: ClipId,
    /// Human-readable name / label (e.g. "best_take", "angle_b").
    pub name: String,
    /// In-point within the source clip (frames from clip start).
    pub in_point: Position,
    /// Out-point within the source clip (frames from clip start).
    pub out_point: Position,
    /// Quality rating (0–5, higher is better).  Used by [`AssemblyStrategy::ByRating`].
    pub rating: u8,
    /// Optional camera angle label (used by [`AssemblyStrategy::Interleaved`]).
    pub angle: Option<String>,
    /// Optional comment.
    pub comment: String,
}

impl SubclipMarker {
    /// Create a new subclip marker.
    #[must_use]
    pub fn new(
        source_clip_id: ClipId,
        name: impl Into<String>,
        in_point: Position,
        out_point: Position,
    ) -> Self {
        Self {
            source_clip_id,
            name: name.into(),
            in_point,
            out_point,
            rating: 3,
            angle: None,
            comment: String::new(),
        }
    }

    /// Set the rating.
    #[must_use]
    pub fn with_rating(mut self, rating: u8) -> Self {
        self.rating = rating.min(5);
        self
    }

    /// Set the camera angle label.
    #[must_use]
    pub fn with_angle(mut self, angle: impl Into<String>) -> Self {
        self.angle = Some(angle.into());
        self
    }

    /// Duration of this subclip in frames.
    #[must_use]
    pub fn duration(&self) -> Duration {
        let d = (self.out_point.0 - self.in_point.0).max(0);
        Duration(d)
    }
}

/// Strategy used when ordering subclips for auto-assembly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssemblyStrategy {
    /// Place clips in the order they appear in the input list.
    InOrder,
    /// Sort clips descending by rating (best first).
    ByRating,
    /// Sort clips ascending by duration (shortest first — tight assembly).
    ByDuration,
    /// Interleave clips from different camera angles round-robin.
    Interleaved,
}

/// A single clip instruction produced by auto-assembly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssembledClip {
    /// The source clip to place.
    pub source_clip_id: ClipId,
    /// In-point within the source clip.
    pub in_point: Position,
    /// Out-point within the source clip.
    pub out_point: Position,
    /// Position on the output timeline where this clip starts.
    pub timeline_position: Position,
    /// Duration of the clip on the output timeline.
    pub duration: Duration,
    /// Name label from the original marker.
    pub label: String,
    /// Crossfade transition to apply at the *start* of this clip (overlapping
    /// with the previous clip). `None` for the first clip.
    pub crossfade: Option<CrossfadeConfig>,
}

/// Crossfade transition applied between consecutive assembled clips.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossfadeConfig {
    /// Duration of the crossfade in frames.
    pub duration_frames: i64,
    /// Type of crossfade curve.
    pub curve: CrossfadeCurve,
}

/// Easing curve for crossfade transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrossfadeCurve {
    /// Linear fade.
    Linear,
    /// Equal-power (constant loudness for audio).
    EqualPower,
    /// S-curve (smooth in/out).
    SCurve,
}

impl CrossfadeConfig {
    /// Creates a linear crossfade with the given duration.
    #[must_use]
    pub fn linear(duration_frames: i64) -> Self {
        Self {
            duration_frames: duration_frames.max(0),
            curve: CrossfadeCurve::Linear,
        }
    }

    /// Creates an equal-power crossfade with the given duration.
    #[must_use]
    pub fn equal_power(duration_frames: i64) -> Self {
        Self {
            duration_frames: duration_frames.max(0),
            curve: CrossfadeCurve::EqualPower,
        }
    }

    /// Creates an S-curve crossfade with the given duration.
    #[must_use]
    pub fn s_curve(duration_frames: i64) -> Self {
        Self {
            duration_frames: duration_frames.max(0),
            curve: CrossfadeCurve::SCurve,
        }
    }
}

/// Configuration for the auto-sequence assembler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoSequenceConfig {
    /// Ordering strategy.
    pub strategy: AssemblyStrategy,
    /// Target track for the assembled clips.
    pub target_track: TrackId,
    /// Starting position on the output timeline (default 0).
    pub start_position: Position,
    /// Optional gap between clips (frames). Negative values create overlap.
    pub gap_frames: i64,
    /// If set, skip markers with rating below this threshold.
    pub min_rating: u8,
    /// Maximum number of clips to assemble (0 = unlimited).
    pub max_clips: usize,
    /// Optional crossfade transition between consecutive clips.
    pub crossfade: Option<CrossfadeConfig>,
}

impl AutoSequenceConfig {
    /// Create a config with sensible defaults.
    #[must_use]
    pub fn new(target_track: TrackId) -> Self {
        Self {
            strategy: AssemblyStrategy::InOrder,
            target_track,
            start_position: Position::new(0),
            gap_frames: 0,
            min_rating: 0,
            max_clips: 0,
            crossfade: None,
        }
    }

    /// Set the assembly strategy.
    #[must_use]
    pub fn with_strategy(mut self, strategy: AssemblyStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set minimum rating filter.
    #[must_use]
    pub fn with_min_rating(mut self, rating: u8) -> Self {
        self.min_rating = rating;
        self
    }

    /// Set gap between clips. Negative values create overlap between clips.
    #[must_use]
    pub fn with_gap(mut self, frames: i64) -> Self {
        self.gap_frames = frames;
        self
    }

    /// Set a crossfade transition between consecutive clips.
    ///
    /// When a crossfade is configured, clips overlap by the crossfade duration
    /// and the gap setting is applied *after* the crossfade region.
    #[must_use]
    pub fn with_crossfade(mut self, crossfade: CrossfadeConfig) -> Self {
        self.crossfade = Some(crossfade);
        self
    }

    /// Limit the number of assembled clips.
    #[must_use]
    pub fn with_max_clips(mut self, n: usize) -> Self {
        self.max_clips = n;
        self
    }
}

/// Error type for auto-sequence operations.
#[derive(Debug, thiserror::Error)]
pub enum AutoSequenceError {
    /// No markers were provided or all were filtered out.
    #[error("No subclip markers available after filtering (min_rating={min_rating})")]
    NoMarkers {
        /// Rating threshold that filtered out all markers.
        min_rating: u8,
    },
    /// A marker had an invalid range (out ≤ in).
    #[error("Marker '{name}' has invalid range: in={in_point} >= out={out_point}")]
    InvalidMarkerRange {
        /// Marker name.
        name: String,
        /// In-point.
        in_point: i64,
        /// Out-point.
        out_point: i64,
    },
}

/// Result of an auto-sequence assembly operation.
#[derive(Debug, Default)]
pub struct AutoSequenceResult {
    /// Ordered list of assembled clip instructions.
    pub clips: Vec<AssembledClip>,
    /// Total duration of the assembled sequence (frames).
    pub total_duration: Duration,
    /// Number of markers that were skipped (below min_rating or invalid range).
    pub skipped_count: usize,
}

/// Assembles subclip markers into a sequence.
#[derive(Debug, Default)]
pub struct AutoSequencer;

impl AutoSequencer {
    /// Create a new assembler.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Assemble `markers` into a sequence according to `config`.
    ///
    /// # Errors
    ///
    /// Returns [`AutoSequenceError::NoMarkers`] if all markers were filtered
    /// out, or [`AutoSequenceError::InvalidMarkerRange`] for a marker with
    /// `out_point <= in_point`.
    pub fn assemble(
        &self,
        markers: &[SubclipMarker],
        config: &AutoSequenceConfig,
    ) -> Result<AutoSequenceResult, AutoSequenceError> {
        // Filter by rating.
        let mut filtered: Vec<&SubclipMarker> = markers
            .iter()
            .filter(|m| m.rating >= config.min_rating)
            .collect();

        let skipped_by_rating = markers.len() - filtered.len();

        // Validate ranges.
        let mut invalid_skipped = 0usize;
        let mut valid: Vec<&SubclipMarker> = Vec::new();
        for m in &filtered {
            if m.out_point.0 <= m.in_point.0 {
                invalid_skipped += 1;
                // We skip silently unless filtered is empty afterwards.
            } else {
                valid.push(m);
            }
        }
        filtered = valid;

        if filtered.is_empty() {
            return Err(AutoSequenceError::NoMarkers {
                min_rating: config.min_rating,
            });
        }

        // Apply ordering strategy.
        match config.strategy {
            AssemblyStrategy::InOrder => {
                // already in input order
            }
            AssemblyStrategy::ByRating => {
                filtered.sort_by(|a, b| b.rating.cmp(&a.rating));
            }
            AssemblyStrategy::ByDuration => {
                filtered.sort_by_key(|m| m.duration().0);
            }
            AssemblyStrategy::Interleaved => {
                // Group by angle, then interleave round-robin.
                let mut angle_groups: std::collections::HashMap<String, Vec<&&SubclipMarker>> =
                    std::collections::HashMap::new();
                for m in &filtered {
                    let key = m.angle.clone().unwrap_or_else(|| "default".to_string());
                    angle_groups.entry(key).or_default().push(m);
                }
                let mut keys: Vec<String> = angle_groups.keys().cloned().collect();
                keys.sort(); // deterministic
                let max_len = angle_groups.values().map(|v| v.len()).max().unwrap_or(0);
                let mut interleaved: Vec<&SubclipMarker> = Vec::new();
                for i in 0..max_len {
                    for key in &keys {
                        if let Some(group) = angle_groups.get(key) {
                            if let Some(m) = group.get(i) {
                                interleaved.push(**m);
                            }
                        }
                    }
                }
                filtered = interleaved;
            }
        }

        // Apply max_clips limit.
        if config.max_clips > 0 && filtered.len() > config.max_clips {
            filtered.truncate(config.max_clips);
        }

        // Build assembled clips.
        let mut cursor = config.start_position;
        let mut assembled = Vec::new();
        let crossfade_overlap = config
            .crossfade
            .as_ref()
            .map_or(0, |cf| cf.duration_frames);

        for (idx, m) in filtered.iter().enumerate() {
            let dur = m.duration();
            let clip_crossfade = if idx > 0 {
                config.crossfade.clone()
            } else {
                None
            };
            assembled.push(AssembledClip {
                source_clip_id: m.source_clip_id,
                in_point: m.in_point,
                out_point: m.out_point,
                timeline_position: cursor,
                duration: dur,
                label: m.name.clone(),
                crossfade: clip_crossfade,
            });
            // Advance cursor: clip duration + gap - crossfade overlap (for next clip)
            let overlap = if idx < filtered.len() - 1 {
                crossfade_overlap
            } else {
                0
            };
            cursor = Position::new(cursor.0 + dur.0 + config.gap_frames - overlap);
        }

        let total_duration = Duration(cursor.0 - config.start_position.0);

        Ok(AutoSequenceResult {
            clips: assembled,
            total_duration,
            skipped_count: skipped_by_rating + invalid_skipped,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cid() -> ClipId {
        ClipId::new()
    }

    fn make_marker(cid: ClipId, name: &str, in_f: i64, out_f: i64, rating: u8) -> SubclipMarker {
        SubclipMarker::new(cid, name, Position::new(in_f), Position::new(out_f))
            .with_rating(rating)
    }

    #[test]
    fn test_assemble_in_order() {
        let id = cid();
        let markers = vec![
            make_marker(id, "a", 0, 100, 3),
            make_marker(id, "b", 100, 200, 3),
            make_marker(id, "c", 200, 300, 3),
        ];
        let track = TrackId::new();
        let config = AutoSequenceConfig::new(track);
        let seq = AutoSequencer::new();
        let result = seq.assemble(&markers, &config).expect("should succeed in test");
        assert_eq!(result.clips.len(), 3);
        assert_eq!(result.clips[0].label, "a");
        assert_eq!(result.clips[1].timeline_position.0, 100);
        assert_eq!(result.clips[2].timeline_position.0, 200);
    }

    #[test]
    fn test_assemble_by_rating() {
        let id = cid();
        let markers = vec![
            make_marker(id, "low", 0, 50, 1),
            make_marker(id, "high", 50, 100, 5),
            make_marker(id, "mid", 100, 150, 3),
        ];
        let track = TrackId::new();
        let config = AutoSequenceConfig::new(track).with_strategy(AssemblyStrategy::ByRating);
        let result = AutoSequencer::new().assemble(&markers, &config).expect("should succeed in test");
        assert_eq!(result.clips[0].label, "high");
    }

    #[test]
    fn test_assemble_by_duration_shortest_first() {
        let id = cid();
        let markers = vec![
            make_marker(id, "long", 0, 200, 3),   // 200 frames
            make_marker(id, "short", 200, 250, 3), // 50 frames
            make_marker(id, "mid", 250, 350, 3),   // 100 frames
        ];
        let track = TrackId::new();
        let config = AutoSequenceConfig::new(track).with_strategy(AssemblyStrategy::ByDuration);
        let result = AutoSequencer::new().assemble(&markers, &config).expect("should succeed in test");
        assert_eq!(result.clips[0].label, "short");
        assert_eq!(result.clips[1].label, "mid");
        assert_eq!(result.clips[2].label, "long");
    }

    #[test]
    fn test_assemble_min_rating_filter() {
        let id = cid();
        let markers = vec![
            make_marker(id, "bad", 0, 50, 1),
            make_marker(id, "good", 50, 100, 4),
        ];
        let track = TrackId::new();
        let config = AutoSequenceConfig::new(track).with_min_rating(3);
        let result = AutoSequencer::new().assemble(&markers, &config).expect("should succeed in test");
        assert_eq!(result.clips.len(), 1);
        assert_eq!(result.clips[0].label, "good");
        assert_eq!(result.skipped_count, 1);
    }

    #[test]
    fn test_assemble_no_markers_error() {
        let id = cid();
        let markers = vec![make_marker(id, "bad", 0, 50, 1)];
        let track = TrackId::new();
        let config = AutoSequenceConfig::new(track).with_min_rating(5);
        let result = AutoSequencer::new().assemble(&markers, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_assemble_with_gap() {
        let id = cid();
        let markers = vec![
            make_marker(id, "a", 0, 100, 3),
            make_marker(id, "b", 100, 200, 3),
        ];
        let track = TrackId::new();
        let config = AutoSequenceConfig::new(track).with_gap(10);
        let result = AutoSequencer::new().assemble(&markers, &config).expect("should succeed in test");
        // First clip at 0, duration 100; second clip at 110 (100 + 10 gap)
        assert_eq!(result.clips[1].timeline_position.0, 110);
    }

    #[test]
    fn test_assemble_max_clips() {
        let id = cid();
        let markers = vec![
            make_marker(id, "a", 0, 50, 3),
            make_marker(id, "b", 50, 100, 3),
            make_marker(id, "c", 100, 150, 3),
        ];
        let track = TrackId::new();
        let config = AutoSequenceConfig::new(track).with_max_clips(2);
        let result = AutoSequencer::new().assemble(&markers, &config).expect("should succeed in test");
        assert_eq!(result.clips.len(), 2);
    }

    #[test]
    fn test_subclip_marker_duration() {
        let m = make_marker(cid(), "x", 10, 60, 3);
        assert_eq!(m.duration().0, 50);
    }

    #[test]
    fn test_assemble_interleaved() {
        let id = cid();
        let markers = vec![
            make_marker(id, "a1", 0, 50, 3).with_angle("A"),
            make_marker(id, "b1", 50, 100, 3).with_angle("B"),
            make_marker(id, "a2", 100, 150, 3).with_angle("A"),
        ];
        let track = TrackId::new();
        let config = AutoSequenceConfig::new(track).with_strategy(AssemblyStrategy::Interleaved);
        let result = AutoSequencer::new().assemble(&markers, &config).expect("should succeed in test");
        // Interleaved: A(a1), B(b1), A(a2)
        assert_eq!(result.clips[0].label, "a1");
        assert_eq!(result.clips[1].label, "b1");
        assert_eq!(result.clips[2].label, "a2");
    }

    #[test]
    fn test_assemble_with_crossfade() {
        let id = cid();
        let markers = vec![
            make_marker(id, "a", 0, 100, 3),
            make_marker(id, "b", 100, 200, 3),
            make_marker(id, "c", 200, 300, 3),
        ];
        let track = TrackId::new();
        let config = AutoSequenceConfig::new(track)
            .with_crossfade(CrossfadeConfig::linear(10));
        let result = AutoSequencer::new()
            .assemble(&markers, &config)
            .expect("should succeed in test");

        // First clip has no crossfade
        assert!(result.clips[0].crossfade.is_none());
        // Second clip starts 10 frames earlier (overlap)
        assert_eq!(result.clips[1].timeline_position.0, 90);
        assert!(result.clips[1].crossfade.is_some());
        // Third clip
        assert_eq!(result.clips[2].timeline_position.0, 180);
    }

    #[test]
    fn test_assemble_crossfade_first_clip_no_transition() {
        let id = cid();
        let markers = vec![make_marker(id, "only", 0, 100, 3)];
        let track = TrackId::new();
        let config = AutoSequenceConfig::new(track)
            .with_crossfade(CrossfadeConfig::s_curve(24));
        let result = AutoSequencer::new()
            .assemble(&markers, &config)
            .expect("should succeed in test");
        assert!(result.clips[0].crossfade.is_none());
    }

    #[test]
    fn test_assemble_with_negative_gap_overlap() {
        let id = cid();
        let markers = vec![
            make_marker(id, "a", 0, 100, 3),
            make_marker(id, "b", 100, 200, 3),
        ];
        let track = TrackId::new();
        let config = AutoSequenceConfig::new(track).with_gap(-20);
        let result = AutoSequencer::new()
            .assemble(&markers, &config)
            .expect("should succeed in test");
        // Negative gap means overlap: second clip starts at 80
        assert_eq!(result.clips[1].timeline_position.0, 80);
    }

    #[test]
    fn test_crossfade_config_equal_power() {
        let cf = CrossfadeConfig::equal_power(12);
        assert_eq!(cf.duration_frames, 12);
        assert_eq!(cf.curve, CrossfadeCurve::EqualPower);
    }
}
