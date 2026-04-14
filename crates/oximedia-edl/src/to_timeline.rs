//! EDL-to-timeline conversion.
//!
//! This module converts EDL events into a simplified timeline model
//! consisting of clips placed on tracks with in/out points.
//! Dissolves are handled by creating overlapping clips on the same track.

use crate::error::{EdlError, EdlResult};
use crate::event::{EditType, EdlEvent};
use crate::timecode::EdlFrameRate;

/// Configuration for EDL-to-timeline conversion.
#[derive(Debug, Clone)]
pub struct EdlToTimelineConfig {
    /// Frame rate to use for the output timeline.
    pub frame_rate: EdlFrameRate,
    /// Whether to create separate tracks for video and audio.
    pub separate_audio_video: bool,
    /// Whether to handle dissolves by creating overlapping clips.
    pub expand_dissolves: bool,
    /// Default track name for video clips when none is specified.
    pub default_video_track: String,
    /// Default track name for audio clips when none is specified.
    pub default_audio_track: String,
}

impl Default for EdlToTimelineConfig {
    fn default() -> Self {
        Self {
            frame_rate: EdlFrameRate::Fps25,
            separate_audio_video: true,
            expand_dissolves: true,
            default_video_track: "V1".to_string(),
            default_audio_track: "A1".to_string(),
        }
    }
}

impl EdlToTimelineConfig {
    /// Create a new config with a specific frame rate.
    #[must_use]
    pub fn with_frame_rate(mut self, frame_rate: EdlFrameRate) -> Self {
        self.frame_rate = frame_rate;
        self
    }

    /// Set whether to separate audio and video tracks.
    #[must_use]
    pub fn with_separate_audio_video(mut self, separate: bool) -> Self {
        self.separate_audio_video = separate;
        self
    }

    /// Set whether to expand dissolves into overlapping clips.
    #[must_use]
    pub fn with_expand_dissolves(mut self, expand: bool) -> Self {
        self.expand_dissolves = expand;
        self
    }
}

/// A clip placed on a timeline track.
#[derive(Debug, Clone, PartialEq)]
pub struct TimelineClip {
    /// Unique identifier for this clip (derived from EDL event number).
    pub id: u32,
    /// Source reel/clip name.
    pub reel: String,
    /// Optional descriptive clip name.
    pub clip_name: Option<String>,
    /// Source in point (frames).
    pub source_in_frames: u64,
    /// Source out point (frames).
    pub source_out_frames: u64,
    /// Record/timeline in point (frames).
    pub timeline_in_frames: u64,
    /// Record/timeline out point (frames).
    pub timeline_out_frames: u64,
    /// The edit type that placed this clip.
    pub edit_type: EditType,
    /// Whether this clip is the outgoing side of a dissolve.
    pub is_dissolve_outgoing: bool,
    /// Whether this clip is the incoming side of a dissolve.
    pub is_dissolve_incoming: bool,
    /// Transition duration in frames (for dissolves/wipes), if any.
    pub transition_frames: Option<u32>,
}

impl TimelineClip {
    /// Duration of this clip on the timeline, in frames.
    #[must_use]
    pub fn duration_frames(&self) -> u64 {
        self.timeline_out_frames
            .saturating_sub(self.timeline_in_frames)
    }

    /// Source duration in frames.
    #[must_use]
    pub fn source_duration_frames(&self) -> u64 {
        self.source_out_frames
            .saturating_sub(self.source_in_frames)
    }
}

/// A named track in the timeline.
#[derive(Debug, Clone)]
pub struct TimelineTrack {
    /// Track name (e.g. "V1", "A1").
    pub name: String,
    /// Whether this is a video track.
    pub is_video: bool,
    /// Clips on this track, ordered by timeline_in_frames.
    pub clips: Vec<TimelineClip>,
}

impl TimelineTrack {
    /// Create a new empty track.
    #[must_use]
    pub fn new(name: impl Into<String>, is_video: bool) -> Self {
        Self {
            name: name.into(),
            is_video,
            clips: Vec::new(),
        }
    }

    /// Total duration covered by clips on this track (max timeline_out - min timeline_in).
    #[must_use]
    pub fn span_frames(&self) -> u64 {
        if self.clips.is_empty() {
            return 0;
        }
        let min_in = self
            .clips
            .iter()
            .map(|c| c.timeline_in_frames)
            .min()
            .unwrap_or(0);
        let max_out = self
            .clips
            .iter()
            .map(|c| c.timeline_out_frames)
            .max()
            .unwrap_or(0);
        max_out.saturating_sub(min_in)
    }

    /// Number of clips on this track.
    #[must_use]
    pub fn clip_count(&self) -> usize {
        self.clips.len()
    }

    /// Sort clips by timeline in point.
    pub fn sort_clips(&mut self) {
        self.clips.sort_by_key(|c| c.timeline_in_frames);
    }
}

/// A complete timeline produced from EDL conversion.
#[derive(Debug, Clone)]
pub struct Timeline {
    /// Title (from the EDL).
    pub title: Option<String>,
    /// Frame rate.
    pub frame_rate: EdlFrameRate,
    /// Tracks in the timeline.
    pub tracks: Vec<TimelineTrack>,
}

impl Timeline {
    /// Total number of clips across all tracks.
    #[must_use]
    pub fn total_clips(&self) -> usize {
        self.tracks.iter().map(|t| t.clip_count()).sum()
    }

    /// Total number of tracks.
    #[must_use]
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Get a track by name.
    #[must_use]
    pub fn get_track(&self, name: &str) -> Option<&TimelineTrack> {
        self.tracks.iter().find(|t| t.name == name)
    }

    /// Get a mutable track by name.
    pub fn get_track_mut(&mut self, name: &str) -> Option<&mut TimelineTrack> {
        self.tracks.iter_mut().find(|t| t.name == name)
    }

    /// Overall timeline duration (max of all track spans).
    #[must_use]
    pub fn duration_frames(&self) -> u64 {
        self.tracks
            .iter()
            .map(|t| {
                t.clips
                    .iter()
                    .map(|c| c.timeline_out_frames)
                    .max()
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0)
    }
}

/// Convert an EDL event list into a `Timeline`.
///
/// # Errors
///
/// Returns `EdlError` if timecode conversion fails.
pub fn convert_to_timeline(
    events: &[EdlEvent],
    title: Option<String>,
    config: &EdlToTimelineConfig,
) -> EdlResult<Timeline> {
    let mut timeline = Timeline {
        title,
        frame_rate: config.frame_rate,
        tracks: Vec::new(),
    };

    for event in events {
        let track_name = determine_track_name(event, config);
        let is_video = event.track.has_video();

        // Ensure track exists
        if timeline.get_track(&track_name).is_none() {
            timeline
                .tracks
                .push(TimelineTrack::new(&track_name, is_video));
        }

        if config.expand_dissolves && event.edit_type == EditType::Dissolve {
            // For a dissolve, the incoming clip extends earlier by the transition duration.
            // We represent this as a normal clip whose timeline-in is pulled back.
            let trans_dur = event.transition_duration.unwrap_or(0) as u64;

            let clip = TimelineClip {
                id: event.number,
                reel: event.reel.clone(),
                clip_name: event.clip_name.clone(),
                source_in_frames: event.source_in.to_frames(),
                source_out_frames: event.source_out.to_frames(),
                timeline_in_frames: event.record_in.to_frames().saturating_sub(trans_dur),
                timeline_out_frames: event.record_out.to_frames(),
                edit_type: event.edit_type,
                is_dissolve_outgoing: false,
                is_dissolve_incoming: true,
                transition_frames: event.transition_duration,
            };

            let track = timeline
                .get_track_mut(&track_name)
                .ok_or_else(|| EdlError::ValidationError("Track not found".to_string()))?;
            track.clips.push(clip);
        } else {
            let clip = TimelineClip {
                id: event.number,
                reel: event.reel.clone(),
                clip_name: event.clip_name.clone(),
                source_in_frames: event.source_in.to_frames(),
                source_out_frames: event.source_out.to_frames(),
                timeline_in_frames: event.record_in.to_frames(),
                timeline_out_frames: event.record_out.to_frames(),
                edit_type: event.edit_type,
                is_dissolve_outgoing: false,
                is_dissolve_incoming: false,
                transition_frames: event.transition_duration,
            };

            let track = timeline
                .get_track_mut(&track_name)
                .ok_or_else(|| EdlError::ValidationError("Track not found".to_string()))?;
            track.clips.push(clip);
        }
    }

    // Sort clips on every track
    for track in &mut timeline.tracks {
        track.sort_clips();
    }

    Ok(timeline)
}

/// Determine which track name an event should be placed on.
fn determine_track_name(event: &EdlEvent, config: &EdlToTimelineConfig) -> String {
    if config.separate_audio_video {
        if event.track.has_video() {
            config.default_video_track.clone()
        } else {
            config.default_audio_track.clone()
        }
    } else {
        // Put everything on the video track
        config.default_video_track.clone()
    }
}

/// Helper to create a simple timeline from an EDL with default config.
///
/// # Errors
///
/// Returns `EdlError` if conversion fails.
pub fn convert_edl_to_timeline(
    edl: &crate::Edl,
) -> EdlResult<Timeline> {
    let config = EdlToTimelineConfig::default()
        .with_frame_rate(edl.frame_rate);
    convert_to_timeline(&edl.events, edl.title.clone(), &config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::AudioChannel;
    use crate::event::{EditType, TrackType};
    use crate::timecode::{EdlFrameRate, EdlTimecode};

    fn tc(h: u8, m: u8, s: u8, f: u8) -> EdlTimecode {
        EdlTimecode::new(h, m, s, f, EdlFrameRate::Fps25).expect("valid timecode")
    }

    fn make_cut(num: u32, reel: &str, rec_in: EdlTimecode, rec_out: EdlTimecode) -> EdlEvent {
        EdlEvent::new(
            num,
            reel.to_string(),
            TrackType::Video,
            EditType::Cut,
            rec_in,
            rec_out,
            rec_in,
            rec_out,
        )
    }

    fn make_dissolve(
        num: u32,
        reel: &str,
        rec_in: EdlTimecode,
        rec_out: EdlTimecode,
        trans_dur: u32,
    ) -> EdlEvent {
        let mut ev = EdlEvent::new(
            num,
            reel.to_string(),
            TrackType::Video,
            EditType::Dissolve,
            rec_in,
            rec_out,
            rec_in,
            rec_out,
        );
        ev.set_transition_duration(trans_dur);
        ev
    }

    #[test]
    fn test_basic_conversion() {
        let events = vec![
            make_cut(1, "A001", tc(1, 0, 0, 0), tc(1, 0, 5, 0)),
            make_cut(2, "A002", tc(1, 0, 5, 0), tc(1, 0, 10, 0)),
        ];
        let config = EdlToTimelineConfig::default();
        let tl = convert_to_timeline(&events, Some("Test".to_string()), &config)
            .expect("conversion should succeed");

        assert_eq!(tl.title.as_deref(), Some("Test"));
        assert_eq!(tl.total_clips(), 2);
        assert_eq!(tl.track_count(), 1);
    }

    #[test]
    fn test_clips_on_correct_track() {
        let events = vec![
            make_cut(1, "A001", tc(1, 0, 0, 0), tc(1, 0, 5, 0)),
        ];
        let config = EdlToTimelineConfig::default();
        let tl = convert_to_timeline(&events, None, &config)
            .expect("conversion should succeed");

        let track = tl.get_track("V1").expect("V1 track should exist");
        assert!(track.is_video);
        assert_eq!(track.clip_count(), 1);
        assert_eq!(track.clips[0].reel, "A001");
    }

    #[test]
    fn test_audio_event_separate_track() {
        let audio_event = EdlEvent::new(
            1,
            "AUD01".to_string(),
            TrackType::Audio(AudioChannel::A1),
            EditType::Cut,
            tc(1, 0, 0, 0),
            tc(1, 0, 5, 0),
            tc(1, 0, 0, 0),
            tc(1, 0, 5, 0),
        );
        let config = EdlToTimelineConfig::default();
        let tl = convert_to_timeline(&[audio_event], None, &config)
            .expect("conversion should succeed");

        assert_eq!(tl.track_count(), 1);
        let track = tl.get_track("A1").expect("A1 track should exist");
        assert!(!track.is_video);
        assert_eq!(track.clip_count(), 1);
    }

    #[test]
    fn test_dissolve_expands_overlap() {
        let events = vec![
            make_cut(1, "A001", tc(1, 0, 0, 0), tc(1, 0, 10, 0)),
            make_dissolve(2, "A002", tc(1, 0, 10, 0), tc(1, 0, 20, 0), 25), // 1-second dissolve at 25fps
        ];
        let config = EdlToTimelineConfig::default();
        let tl = convert_to_timeline(&events, None, &config)
            .expect("conversion should succeed");

        let track = tl.get_track("V1").expect("V1 track should exist");
        assert_eq!(track.clip_count(), 2);

        // The dissolve clip should have its timeline_in pulled back by 25 frames
        let dissolve_clip = &track.clips[1];
        assert!(dissolve_clip.is_dissolve_incoming);
        let expected_in = tc(1, 0, 10, 0).to_frames() - 25;
        assert_eq!(dissolve_clip.timeline_in_frames, expected_in);
    }

    #[test]
    fn test_dissolve_without_expansion() {
        let events = vec![
            make_dissolve(1, "A001", tc(1, 0, 0, 0), tc(1, 0, 10, 0), 25),
        ];
        let config = EdlToTimelineConfig::default().with_expand_dissolves(false);
        let tl = convert_to_timeline(&events, None, &config)
            .expect("conversion should succeed");

        let track = tl.get_track("V1").expect("V1 track should exist");
        let clip = &track.clips[0];
        assert!(!clip.is_dissolve_incoming);
        // Timeline in should NOT be pulled back
        assert_eq!(clip.timeline_in_frames, tc(1, 0, 0, 0).to_frames());
    }

    #[test]
    fn test_mixed_video_and_audio_separate() {
        let video = make_cut(1, "V01", tc(1, 0, 0, 0), tc(1, 0, 5, 0));
        let audio = EdlEvent::new(
            2,
            "AUD01".to_string(),
            TrackType::Audio(AudioChannel::A1),
            EditType::Cut,
            tc(1, 0, 0, 0),
            tc(1, 0, 5, 0),
            tc(1, 0, 0, 0),
            tc(1, 0, 5, 0),
        );
        let config = EdlToTimelineConfig::default();
        let tl = convert_to_timeline(&[video, audio], None, &config)
            .expect("conversion should succeed");

        assert_eq!(tl.track_count(), 2);
        assert!(tl.get_track("V1").is_some());
        assert!(tl.get_track("A1").is_some());
    }

    #[test]
    fn test_mixed_without_separate() {
        let video = make_cut(1, "V01", tc(1, 0, 0, 0), tc(1, 0, 5, 0));
        let audio = EdlEvent::new(
            2,
            "AUD01".to_string(),
            TrackType::Audio(AudioChannel::A1),
            EditType::Cut,
            tc(1, 0, 0, 0),
            tc(1, 0, 5, 0),
            tc(1, 0, 0, 0),
            tc(1, 0, 5, 0),
        );
        let config = EdlToTimelineConfig::default().with_separate_audio_video(false);
        let tl = convert_to_timeline(&[video, audio], None, &config)
            .expect("conversion should succeed");

        assert_eq!(tl.track_count(), 1);
        assert_eq!(tl.total_clips(), 2);
    }

    #[test]
    fn test_clip_duration() {
        let clip = TimelineClip {
            id: 1,
            reel: "A001".to_string(),
            clip_name: None,
            source_in_frames: 0,
            source_out_frames: 125,
            timeline_in_frames: 100,
            timeline_out_frames: 225,
            edit_type: EditType::Cut,
            is_dissolve_outgoing: false,
            is_dissolve_incoming: false,
            transition_frames: None,
        };
        assert_eq!(clip.duration_frames(), 125);
        assert_eq!(clip.source_duration_frames(), 125);
    }

    #[test]
    fn test_timeline_duration() {
        let events = vec![
            make_cut(1, "A001", tc(1, 0, 0, 0), tc(1, 0, 5, 0)),
            make_cut(2, "A002", tc(1, 0, 5, 0), tc(1, 0, 10, 0)),
        ];
        let config = EdlToTimelineConfig::default();
        let tl = convert_to_timeline(&events, None, &config)
            .expect("conversion should succeed");

        let expected_dur = tc(1, 0, 10, 0).to_frames();
        assert_eq!(tl.duration_frames(), expected_dur);
    }

    #[test]
    fn test_track_span() {
        let events = vec![
            make_cut(1, "A001", tc(1, 0, 0, 0), tc(1, 0, 5, 0)),
            make_cut(2, "A002", tc(1, 0, 10, 0), tc(1, 0, 15, 0)),
        ];
        let config = EdlToTimelineConfig::default();
        let tl = convert_to_timeline(&events, None, &config)
            .expect("conversion should succeed");

        let track = tl.get_track("V1").expect("V1 should exist");
        let span = track.span_frames();
        let expected = tc(1, 0, 15, 0).to_frames() - tc(1, 0, 0, 0).to_frames();
        assert_eq!(span, expected);
    }

    #[test]
    fn test_empty_events() {
        let config = EdlToTimelineConfig::default();
        let tl = convert_to_timeline(&[], None, &config)
            .expect("conversion should succeed");
        assert_eq!(tl.total_clips(), 0);
        assert_eq!(tl.track_count(), 0);
        assert_eq!(tl.duration_frames(), 0);
    }

    #[test]
    fn test_convert_edl_helper() {
        let mut edl = crate::Edl::new(crate::EdlFormat::Cmx3600);
        edl.set_frame_rate(EdlFrameRate::Fps25);
        edl.set_title("Helper Test".to_string());

        let ev = make_cut(1, "A001", tc(1, 0, 0, 0), tc(1, 0, 5, 0));
        edl.add_event(ev).expect("add_event should succeed");

        let tl = convert_edl_to_timeline(&edl).expect("conversion should succeed");
        assert_eq!(tl.title.as_deref(), Some("Helper Test"));
        assert_eq!(tl.total_clips(), 1);
    }

    #[test]
    fn test_clip_name_propagated() {
        let mut ev = make_cut(1, "A001", tc(1, 0, 0, 0), tc(1, 0, 5, 0));
        ev.set_clip_name("interview.mov".to_string());

        let config = EdlToTimelineConfig::default();
        let tl = convert_to_timeline(&[ev], None, &config)
            .expect("conversion should succeed");

        let track = tl.get_track("V1").expect("V1 should exist");
        assert_eq!(
            track.clips[0].clip_name.as_deref(),
            Some("interview.mov")
        );
    }
}
