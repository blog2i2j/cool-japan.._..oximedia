//! Routing automation with timecode support.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::curves::{CurveType, GainAutomation, GainCurveSegment};

/// Timecode representation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Timecode {
    /// Hours (0-23)
    pub hours: u8,
    /// Minutes (0-59)
    pub minutes: u8,
    /// Seconds (0-59)
    pub seconds: u8,
    /// Frames (0-frame_rate-1)
    pub frames: u8,
    /// Frame rate
    pub frame_rate: FrameRate,
}

/// Frame rate for timecode
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FrameRate {
    /// 24 fps (film)
    Fps24,
    /// 25 fps (PAL)
    Fps25,
    /// 29.97 fps drop-frame (NTSC)
    Fps2997Df,
    /// 29.97 fps non-drop (NTSC)
    Fps2997Ndf,
    /// 30 fps
    Fps30,
    /// 50 fps
    Fps50,
    /// 59.94 fps
    Fps5994,
    /// 60 fps
    Fps60,
}

impl FrameRate {
    /// Get the numeric frame rate
    #[must_use]
    pub const fn as_u8(&self) -> u8 {
        match self {
            Self::Fps24 => 24,
            Self::Fps25 => 25,
            Self::Fps2997Df | Self::Fps2997Ndf => 30,
            Self::Fps30 => 30,
            Self::Fps50 => 50,
            Self::Fps5994 => 60,
            Self::Fps60 => 60,
        }
    }
}

impl Timecode {
    /// Create a new timecode
    #[must_use]
    pub const fn new(
        hours: u8,
        minutes: u8,
        seconds: u8,
        frames: u8,
        frame_rate: FrameRate,
    ) -> Self {
        Self {
            hours,
            minutes,
            seconds,
            frames,
            frame_rate,
        }
    }

    /// Create timecode from total frames
    #[must_use]
    pub fn from_frames(total_frames: u64, frame_rate: FrameRate) -> Self {
        let fps = u64::from(frame_rate.as_u8());
        let frames = (total_frames % fps) as u8;
        let total_seconds = total_frames / fps;
        let seconds = (total_seconds % 60) as u8;
        let total_minutes = total_seconds / 60;
        let minutes = (total_minutes % 60) as u8;
        let hours = (total_minutes / 60) as u8;

        Self {
            hours,
            minutes,
            seconds,
            frames,
            frame_rate,
        }
    }

    /// Convert to total frames
    #[must_use]
    pub fn to_frames(&self) -> u64 {
        let fps = u64::from(self.frame_rate.as_u8());
        u64::from(self.hours) * 3600 * fps
            + u64::from(self.minutes) * 60 * fps
            + u64::from(self.seconds) * fps
            + u64::from(self.frames)
    }
}

/// Routing automation action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AutomationAction {
    /// Connect two points
    Connect {
        source: usize,
        destination: usize,
        gain_db: f32,
    },
    /// Disconnect two points
    Disconnect {
        /// Source index
        source: usize,
        /// Destination index
        destination: usize,
    },
    /// Set gain
    SetGain {
        /// Channel index
        channel: usize,
        /// Gain in dB
        gain_db: f32,
    },
    /// Mute channel
    Mute {
        /// Channel index
        channel: usize,
    },
    /// Unmute channel
    Unmute {
        /// Channel index
        channel: usize,
    },
    /// Load preset
    LoadPreset {
        /// Preset ID
        preset_id: u64,
    },
    /// Custom action
    Custom {
        /// Action type identifier
        action_type: String,
        /// Action parameters
        parameters: Vec<f32>,
    },
}

/// Automation event at a specific timecode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationEvent {
    /// Timecode when this event triggers
    pub timecode: Timecode,
    /// Action to perform
    pub action: AutomationAction,
    /// Event description
    pub description: String,
    /// Whether this event is enabled
    pub enabled: bool,
}

/// Automation timeline with discrete events and continuous curve-based gain
/// automation.
///
/// Discrete routing events (connect, disconnect, mute, etc.) are stored in the
/// `events` BTreeMap and retrieved by exact timecode.  Smooth gain transitions
/// (linear fades, S-curves, exponential fades) are managed by the embedded
/// [`GainAutomation`] and evaluated at arbitrary frame positions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationTimeline {
    /// Events indexed by timecode
    events: BTreeMap<Timecode, Vec<AutomationEvent>>,
    /// Timeline name
    pub name: String,
    /// Frame rate for this timeline
    pub frame_rate: FrameRate,
    /// Curve-based gain automation integrated into this timeline.
    gain_automation: GainAutomation,
}

impl AutomationTimeline {
    /// Create a new automation timeline
    #[must_use]
    pub fn new(name: String, frame_rate: FrameRate) -> Self {
        Self {
            events: BTreeMap::new(),
            name,
            frame_rate,
            gain_automation: GainAutomation::new(),
        }
    }

    /// Add an event
    pub fn add_event(&mut self, event: AutomationEvent) {
        self.events.entry(event.timecode).or_default().push(event);
    }

    /// Remove events at a timecode
    pub fn remove_events_at(&mut self, timecode: Timecode) {
        self.events.remove(&timecode);
    }

    /// Get events at a specific timecode
    #[must_use]
    pub fn get_events_at(&self, timecode: Timecode) -> Option<&Vec<AutomationEvent>> {
        self.events.get(&timecode)
    }

    /// Get all events in a time range
    #[must_use]
    pub fn get_events_in_range(&self, start: Timecode, end: Timecode) -> Vec<&AutomationEvent> {
        self.events
            .range(start..=end)
            .flat_map(|(_, events)| events)
            .collect()
    }

    /// Get all enabled events
    #[must_use]
    pub fn get_enabled_events(&self) -> Vec<&AutomationEvent> {
        self.events
            .values()
            .flatten()
            .filter(|e| e.enabled)
            .collect()
    }

    /// Get total event count
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.events.values().map(Vec::len).sum()
    }

    /// Clear all events
    pub fn clear(&mut self) {
        self.events.clear();
    }

    // -----------------------------------------------------------------------
    // Curve-based gain automation
    // -----------------------------------------------------------------------

    /// Adds a raw gain automation curve segment to this timeline.
    ///
    /// Frame positions use the same absolute frame numbering as
    /// [`Timecode::to_frames`].  Segments are kept sorted by start frame so
    /// evaluation is always consistent.
    pub fn add_gain_segment(&mut self, segment: GainCurveSegment) {
        self.gain_automation.add_segment(segment);
    }

    /// Evaluates the gain automation at the given absolute frame for a channel.
    ///
    /// Returns `None` if no curve segment covers the frame.
    #[must_use]
    pub fn gain_at_frame(&self, frame: u64, channel: Option<usize>) -> Option<f32> {
        self.gain_automation.evaluate(frame, channel)
    }

    /// Evaluates the gain automation at the given timecode for a channel.
    ///
    /// Converts the timecode to frames using [`Timecode::to_frames`] then
    /// delegates to [`Self::gain_at_frame`].
    #[must_use]
    pub fn gain_at_timecode(&self, tc: Timecode, channel: Option<usize>) -> Option<f32> {
        self.gain_automation.evaluate(tc.to_frames(), channel)
    }

    /// Schedules a linear gain fade from `from_db` to `to_db` between the
    /// given timecodes, optionally on a specific channel.
    pub fn add_linear_fade(
        &mut self,
        start_tc: Timecode,
        end_tc: Timecode,
        from_db: f32,
        to_db: f32,
        channel: Option<usize>,
    ) {
        let seg = GainCurveSegment::new(
            start_tc.to_frames(),
            end_tc.to_frames(),
            from_db,
            to_db,
            CurveType::Linear,
        );
        let seg = if let Some(ch) = channel {
            seg.with_channel(ch)
        } else {
            seg
        };
        self.gain_automation.add_segment(seg);
    }

    /// Schedules an S-curve (smoothstep) gain transition from `from_db` to
    /// `to_db` between the given timecodes, optionally on a specific channel.
    pub fn add_scurve_fade(
        &mut self,
        start_tc: Timecode,
        end_tc: Timecode,
        from_db: f32,
        to_db: f32,
        channel: Option<usize>,
    ) {
        let seg = GainCurveSegment::new(
            start_tc.to_frames(),
            end_tc.to_frames(),
            from_db,
            to_db,
            CurveType::SCurve,
        );
        let seg = if let Some(ch) = channel {
            seg.with_channel(ch)
        } else {
            seg
        };
        self.gain_automation.add_segment(seg);
    }

    /// Schedules an exponential fade-in to `target_db` between `start_tc`
    /// and `end_tc`, optionally on a specific channel.
    pub fn add_fade_in(
        &mut self,
        start_tc: Timecode,
        end_tc: Timecode,
        target_db: f32,
        channel: Option<usize>,
    ) {
        let mut seg = GainCurveSegment::new(
            start_tc.to_frames(),
            end_tc.to_frames(),
            f32::NEG_INFINITY,
            target_db,
            CurveType::ExponentialOut,
        );
        if let Some(ch) = channel {
            seg = seg.with_channel(ch);
        }
        self.gain_automation.add_segment(seg);
    }

    /// Schedules an exponential fade-out from `from_db` between `start_tc`
    /// and `end_tc`, optionally on a specific channel.
    pub fn add_fade_out(
        &mut self,
        start_tc: Timecode,
        end_tc: Timecode,
        from_db: f32,
        channel: Option<usize>,
    ) {
        let mut seg = GainCurveSegment::new(
            start_tc.to_frames(),
            end_tc.to_frames(),
            from_db,
            f32::NEG_INFINITY,
            CurveType::ExponentialIn,
        );
        if let Some(ch) = channel {
            seg = seg.with_channel(ch);
        }
        self.gain_automation.add_segment(seg);
    }

    /// Returns the number of gain automation curve segments in this timeline.
    #[must_use]
    pub fn gain_segment_count(&self) -> usize {
        self.gain_automation.segment_count()
    }

    /// Returns a reference to the underlying [`GainAutomation`].
    #[must_use]
    pub fn gain_automation(&self) -> &GainAutomation {
        &self.gain_automation
    }

    /// Removes all gain automation segments that end before the given frame.
    pub fn prune_gain_before(&mut self, frame: u64) {
        self.gain_automation.prune_before(frame);
    }
}

#[cfg(test)]
mod tests {
    use super::super::curves::{CurveType as CT, GainCurveSegment};
    use super::*;

    #[test]
    fn test_timecode_creation() {
        let tc = Timecode::new(1, 30, 45, 12, FrameRate::Fps25);
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 30);
        assert_eq!(tc.seconds, 45);
        assert_eq!(tc.frames, 12);
    }

    #[test]
    fn test_timecode_from_frames() {
        let tc = Timecode::from_frames(150, FrameRate::Fps25);
        assert_eq!(tc.seconds, 6);
        assert_eq!(tc.frames, 0);
    }

    #[test]
    fn test_timecode_to_frames() {
        let tc = Timecode::new(0, 0, 10, 0, FrameRate::Fps25);
        assert_eq!(tc.to_frames(), 250);
    }

    #[test]
    fn test_timeline_creation() {
        let timeline = AutomationTimeline::new("Show 1".to_string(), FrameRate::Fps25);
        assert_eq!(timeline.event_count(), 0);
    }

    #[test]
    fn test_add_event() {
        let mut timeline = AutomationTimeline::new("Test".to_string(), FrameRate::Fps25);

        let event = AutomationEvent {
            timecode: Timecode::new(0, 1, 0, 0, FrameRate::Fps25),
            action: AutomationAction::Mute { channel: 0 },
            description: "Mute channel 0".to_string(),
            enabled: true,
        };

        timeline.add_event(event);
        assert_eq!(timeline.event_count(), 1);
    }

    #[test]
    fn test_get_events_at() {
        let mut timeline = AutomationTimeline::new("Test".to_string(), FrameRate::Fps25);

        let tc = Timecode::new(0, 1, 0, 0, FrameRate::Fps25);
        let event = AutomationEvent {
            timecode: tc,
            action: AutomationAction::Mute { channel: 0 },
            description: "Test".to_string(),
            enabled: true,
        };

        timeline.add_event(event);

        let events = timeline.get_events_at(tc);
        assert!(events.is_some());
        assert_eq!(events.expect("should succeed in test").len(), 1);
    }

    #[test]
    fn test_get_events_in_range() {
        let mut timeline = AutomationTimeline::new("Test".to_string(), FrameRate::Fps25);

        let tc1 = Timecode::new(0, 1, 0, 0, FrameRate::Fps25);
        let tc2 = Timecode::new(0, 2, 0, 0, FrameRate::Fps25);
        let tc3 = Timecode::new(0, 3, 0, 0, FrameRate::Fps25);

        timeline.add_event(AutomationEvent {
            timecode: tc1,
            action: AutomationAction::Mute { channel: 0 },
            description: "Event 1".to_string(),
            enabled: true,
        });

        timeline.add_event(AutomationEvent {
            timecode: tc2,
            action: AutomationAction::Mute { channel: 1 },
            description: "Event 2".to_string(),
            enabled: true,
        });

        timeline.add_event(AutomationEvent {
            timecode: tc3,
            action: AutomationAction::Mute { channel: 2 },
            description: "Event 3".to_string(),
            enabled: true,
        });

        let events = timeline.get_events_in_range(tc1, tc2);
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_remove_events() {
        let mut timeline = AutomationTimeline::new("Test".to_string(), FrameRate::Fps25);

        let tc = Timecode::new(0, 1, 0, 0, FrameRate::Fps25);
        timeline.add_event(AutomationEvent {
            timecode: tc,
            action: AutomationAction::Mute { channel: 0 },
            description: "Test".to_string(),
            enabled: true,
        });

        assert_eq!(timeline.event_count(), 1);

        timeline.remove_events_at(tc);
        assert_eq!(timeline.event_count(), 0);
    }

    #[test]
    fn test_enabled_events() {
        let mut timeline = AutomationTimeline::new("Test".to_string(), FrameRate::Fps25);

        timeline.add_event(AutomationEvent {
            timecode: Timecode::new(0, 1, 0, 0, FrameRate::Fps25),
            action: AutomationAction::Mute { channel: 0 },
            description: "Enabled".to_string(),
            enabled: true,
        });

        timeline.add_event(AutomationEvent {
            timecode: Timecode::new(0, 2, 0, 0, FrameRate::Fps25),
            action: AutomationAction::Mute { channel: 1 },
            description: "Disabled".to_string(),
            enabled: false,
        });

        let enabled = timeline.get_enabled_events();
        assert_eq!(enabled.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Curve-based gain automation integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_gain_segment_count_starts_zero() {
        let timeline = AutomationTimeline::new("Test".to_string(), FrameRate::Fps25);
        assert_eq!(timeline.gain_segment_count(), 0);
    }

    #[test]
    fn test_add_gain_segment_increases_count() {
        let mut timeline = AutomationTimeline::new("Test".to_string(), FrameRate::Fps25);
        let seg = GainCurveSegment::new(0, 100, 0.0, -6.0, CT::Linear);
        timeline.add_gain_segment(seg);
        assert_eq!(timeline.gain_segment_count(), 1);
    }

    #[test]
    fn test_gain_at_frame_midpoint_linear() {
        let mut timeline = AutomationTimeline::new("Test".to_string(), FrameRate::Fps25);
        // 0 dB at frame 0 → -20 dB at frame 100 (linear)
        let seg = GainCurveSegment::new(0, 100, 0.0, -20.0, CT::Linear);
        timeline.add_gain_segment(seg);
        let g = timeline.gain_at_frame(50, None).expect("should be Some");
        assert!(
            (g - (-10.0)).abs() < 0.1,
            "expected -10 dB at midpoint, got {g}"
        );
    }

    #[test]
    fn test_gain_at_frame_returns_none_outside_segment() {
        let mut timeline = AutomationTimeline::new("Test".to_string(), FrameRate::Fps25);
        let seg = GainCurveSegment::new(100, 200, 0.0, -6.0, CT::Linear);
        timeline.add_gain_segment(seg);
        assert!(timeline.gain_at_frame(50, None).is_none());
    }

    #[test]
    fn test_add_linear_fade_via_timecode() {
        let mut timeline = AutomationTimeline::new("Test".to_string(), FrameRate::Fps25);
        // 1 second = 25 frames at 25fps
        let start_tc = Timecode::new(0, 0, 0, 0, FrameRate::Fps25);
        let end_tc = Timecode::new(0, 0, 1, 0, FrameRate::Fps25);
        timeline.add_linear_fade(start_tc, end_tc, 0.0, -20.0, None);
        assert_eq!(timeline.gain_segment_count(), 1);
        // Check midpoint at frame 12
        let g = timeline
            .gain_at_frame(12, None)
            .expect("should be inside segment");
        // frame 12 of 25 → gain between 0 and -20
        assert!(g < 0.0 && g > -20.0, "expected partial fade, got {g}");
    }

    #[test]
    fn test_add_scurve_fade_via_timecode() {
        let mut timeline = AutomationTimeline::new("Test".to_string(), FrameRate::Fps25);
        let start_tc = Timecode::new(0, 0, 0, 0, FrameRate::Fps25);
        let end_tc = Timecode::new(0, 0, 2, 0, FrameRate::Fps25);
        timeline.add_scurve_fade(start_tc, end_tc, 0.0, -40.0, Some(0));
        assert_eq!(timeline.gain_segment_count(), 1);
        // S-curve at midpoint (frame 25 of 50) should equal linear midpoint (-20)
        let g = timeline.gain_at_frame(25, Some(0)).expect("should be Some");
        assert!(
            (g - (-20.0)).abs() < 0.5,
            "S-curve midpoint should ≈ -20 dB, got {g}"
        );
    }

    #[test]
    fn test_gain_at_timecode_evaluates_correctly() {
        let mut timeline = AutomationTimeline::new("Test".to_string(), FrameRate::Fps25);
        // Segment spans frames 0..=100
        let seg = GainCurveSegment::new(0, 100, 0.0, -10.0, CT::Linear);
        timeline.add_gain_segment(seg);
        // Timecode 0:00:02:00 at 25fps = frame 50
        let tc = Timecode::new(0, 0, 2, 0, FrameRate::Fps25);
        let g = timeline.gain_at_timecode(tc, None);
        // Frame 50 of 100 → 50% → -5 dB
        assert!(g.is_some());
        assert!((g.expect("Some") - (-5.0)).abs() < 0.1);
    }

    #[test]
    fn test_add_fade_in_creates_segment() {
        let mut timeline = AutomationTimeline::new("Test".to_string(), FrameRate::Fps25);
        let start_tc = Timecode::new(0, 0, 0, 0, FrameRate::Fps25);
        let end_tc = Timecode::new(0, 0, 1, 0, FrameRate::Fps25);
        timeline.add_fade_in(start_tc, end_tc, 0.0, None);
        assert_eq!(timeline.gain_segment_count(), 1);
        let seg = &timeline.gain_automation().segments()[0];
        assert_eq!(seg.curve_type, CT::ExponentialOut);
    }

    #[test]
    fn test_add_fade_out_creates_segment() {
        let mut timeline = AutomationTimeline::new("Test".to_string(), FrameRate::Fps25);
        let start_tc = Timecode::new(0, 0, 0, 0, FrameRate::Fps25);
        let end_tc = Timecode::new(0, 0, 1, 0, FrameRate::Fps25);
        timeline.add_fade_out(start_tc, end_tc, 0.0, None);
        assert_eq!(timeline.gain_segment_count(), 1);
        let seg = &timeline.gain_automation().segments()[0];
        assert_eq!(seg.curve_type, CT::ExponentialIn);
    }

    #[test]
    fn test_prune_gain_before() {
        let mut timeline = AutomationTimeline::new("Test".to_string(), FrameRate::Fps25);
        timeline.add_gain_segment(GainCurveSegment::new(0, 50, 0.0, -6.0, CT::Linear));
        timeline.add_gain_segment(GainCurveSegment::new(100, 200, -6.0, -12.0, CT::Linear));
        timeline.prune_gain_before(60);
        assert_eq!(timeline.gain_segment_count(), 1);
    }

    #[test]
    fn test_gain_automation_independent_of_events() {
        let mut timeline = AutomationTimeline::new("Test".to_string(), FrameRate::Fps25);
        timeline.add_event(AutomationEvent {
            timecode: Timecode::new(0, 0, 5, 0, FrameRate::Fps25),
            action: AutomationAction::Mute { channel: 0 },
            description: "Mute".to_string(),
            enabled: true,
        });
        timeline.add_gain_segment(GainCurveSegment::new(0, 200, 0.0, -20.0, CT::Linear));
        // Events and gain segments should not interfere
        assert_eq!(timeline.event_count(), 1);
        assert_eq!(timeline.gain_segment_count(), 1);
    }

    #[test]
    fn test_channel_specific_gain_fade() {
        let mut timeline = AutomationTimeline::new("Test".to_string(), FrameRate::Fps25);
        // Channel 0 fades from 0 to -20
        timeline.add_gain_segment(
            GainCurveSegment::new(0, 100, 0.0, -20.0, CT::Linear).with_channel(0),
        );
        // Channel 1 fades from 0 to -10
        timeline.add_gain_segment(
            GainCurveSegment::new(0, 100, 0.0, -10.0, CT::Linear).with_channel(1),
        );
        let g0 = timeline.gain_at_frame(50, Some(0)).expect("ch0 Some");
        let g1 = timeline.gain_at_frame(50, Some(1)).expect("ch1 Some");
        assert!((g0 - (-10.0)).abs() < 0.1, "ch0 midpoint -10, got {g0}");
        assert!((g1 - (-5.0)).abs() < 0.1, "ch1 midpoint -5, got {g1}");
    }

    #[test]
    fn test_multiple_segments_sorted_order() {
        let mut timeline = AutomationTimeline::new("Test".to_string(), FrameRate::Fps25);
        // Add in reverse order
        timeline.add_gain_segment(GainCurveSegment::new(200, 300, -12.0, -20.0, CT::Linear));
        timeline.add_gain_segment(GainCurveSegment::new(0, 100, 0.0, -6.0, CT::Linear));
        let segs = timeline.gain_automation().segments();
        assert_eq!(segs.len(), 2);
        assert!(
            segs[0].start_frame <= segs[1].start_frame,
            "segments should be sorted"
        );
    }

    // ── Sub-frame accuracy at various frame rates ────────────────────────────

    /// Verify that `Timecode::to_frames` / `from_frames` round-trips correctly
    /// at every supported frame rate.
    #[test]
    fn test_timecode_round_trip_all_frame_rates() {
        let test_cases: &[(FrameRate, u8)] = &[
            (FrameRate::Fps24, 24),
            (FrameRate::Fps25, 25),
            (FrameRate::Fps2997Ndf, 30),
            (FrameRate::Fps2997Df, 30),
            (FrameRate::Fps30, 30),
            (FrameRate::Fps50, 50),
            (FrameRate::Fps5994, 60),
            (FrameRate::Fps60, 60),
        ];

        for &(fr, fps) in test_cases {
            // One full second at each frame rate.
            let tc = Timecode::new(0, 0, 1, 0, fr);
            let frames = tc.to_frames();
            assert_eq!(
                frames,
                u64::from(fps),
                "{fr:?}: 0:00:01:00 should be {fps} frames, got {frames}"
            );

            // Round-trip: frames → Timecode → frames
            let tc2 = Timecode::from_frames(frames, fr);
            let frames2 = tc2.to_frames();
            assert_eq!(
                frames2, frames,
                "{fr:?}: round-trip should preserve frame count"
            );

            // The last frame of a second (frame fps-1) must decode back correctly.
            let last_frame_tc = Timecode::new(0, 0, 0, fps - 1, fr);
            let f = last_frame_tc.to_frames();
            assert_eq!(
                f,
                u64::from(fps) - 1,
                "{fr:?}: frame {} at 0:00:00:{} should be {} total",
                fps - 1,
                fps - 1,
                fps - 1
            );
        }
    }

    /// At 24 fps the gain at the midpoint of a 24-frame (1 second) segment
    /// should be -10 dB when fading from 0 dB to -20 dB linearly.
    #[test]
    fn test_gain_at_frame_midpoint_24fps_linear() {
        let mut timeline = AutomationTimeline::new("24fps Test".to_string(), FrameRate::Fps24);
        // 1 second at 24fps = frames 0..24
        let seg = GainCurveSegment::new(0, 24, 0.0, -20.0, CT::Linear);
        timeline.add_gain_segment(seg);
        let g = timeline
            .gain_at_frame(12, None)
            .expect("midpoint inside segment");
        assert!(
            (g - (-10.0)).abs() < 0.5,
            "24fps linear fade midpoint should be ≈ -10 dB, got {g}"
        );
    }

    /// At 30 fps the gain at the midpoint of a 30-frame segment should be -10 dB.
    #[test]
    fn test_gain_at_frame_midpoint_30fps_linear() {
        let mut timeline = AutomationTimeline::new("30fps Test".to_string(), FrameRate::Fps30);
        let seg = GainCurveSegment::new(0, 30, 0.0, -20.0, CT::Linear);
        timeline.add_gain_segment(seg);
        let g = timeline
            .gain_at_frame(15, None)
            .expect("midpoint inside segment");
        assert!(
            (g - (-10.0)).abs() < 0.5,
            "30fps linear fade midpoint should be ≈ -10 dB, got {g}"
        );
    }

    /// At 60 fps the gain at the midpoint of a 60-frame segment should be -10 dB.
    #[test]
    fn test_gain_at_frame_midpoint_60fps_linear() {
        let mut timeline = AutomationTimeline::new("60fps Test".to_string(), FrameRate::Fps60);
        let seg = GainCurveSegment::new(0, 60, 0.0, -20.0, CT::Linear);
        timeline.add_gain_segment(seg);
        let g = timeline
            .gain_at_frame(30, None)
            .expect("midpoint inside segment");
        assert!(
            (g - (-10.0)).abs() < 0.5,
            "60fps linear fade midpoint should be ≈ -10 dB, got {g}"
        );
    }

    /// `gain_at_timecode` at 24 fps: 1 second into a 2-second linear fade
    /// from 0 dB to -20 dB should yield ≈ -10 dB.
    #[test]
    fn test_gain_at_timecode_24fps_midpoint() {
        let mut timeline = AutomationTimeline::new("24fps TC Test".to_string(), FrameRate::Fps24);
        // Fade over 2 seconds = 48 frames
        let seg = GainCurveSegment::new(0, 48, 0.0, -20.0, CT::Linear);
        timeline.add_gain_segment(seg);

        // 1:00 at 24fps = frame 24
        let tc = Timecode::new(0, 0, 1, 0, FrameRate::Fps24);
        let g = timeline
            .gain_at_timecode(tc, None)
            .expect("timecode inside segment");
        assert!(
            (g - (-10.0)).abs() < 0.5,
            "24fps: gain at 0:00:01:00 (frame 24 of 48) should be ≈ -10 dB, got {g}"
        );
    }

    /// `gain_at_timecode` at 50 fps: sub-frame boundary check.
    /// A segment spanning frames 0–50 (1 second) evaluated at frame 25 should
    /// yield exactly the linear midpoint.
    #[test]
    fn test_gain_at_timecode_50fps_midpoint() {
        let mut timeline = AutomationTimeline::new("50fps TC Test".to_string(), FrameRate::Fps50);
        let seg = GainCurveSegment::new(0, 50, 0.0, -10.0, CT::Linear);
        timeline.add_gain_segment(seg);

        // 0.5 seconds at 50fps = frame 25
        let tc = Timecode::new(0, 0, 0, 25, FrameRate::Fps50);
        let g = timeline
            .gain_at_timecode(tc, None)
            .expect("frame 25 inside segment");
        assert!(
            (g - (-5.0)).abs() < 0.2,
            "50fps: gain at frame 25 of 50 should be ≈ -5 dB, got {g}"
        );
    }

    /// Verify that a linear fade evaluates correctly at the very first and last
    /// frames across multiple frame rates.
    #[test]
    fn test_gain_at_first_and_last_frame_various_fps() {
        let fps_list: &[(FrameRate, u64)] = &[
            (FrameRate::Fps24, 24),
            (FrameRate::Fps25, 25),
            (FrameRate::Fps30, 30),
            (FrameRate::Fps50, 50),
            (FrameRate::Fps60, 60),
        ];
        for &(fr, fps) in fps_list {
            let mut timeline = AutomationTimeline::new("boundary test".to_string(), fr);
            // Inclusive [0, fps]: start=0dB, end=-20dB
            let seg = GainCurveSegment::new(0, fps, 0.0, -20.0, CT::Linear);
            timeline.add_gain_segment(seg);

            let g_start = timeline
                .gain_at_frame(0, None)
                .expect("frame 0 should be inside segment");
            assert!(
                g_start.abs() < 0.5,
                "{fr:?}: gain at frame 0 should be ≈ 0 dB, got {g_start}"
            );

            let g_end = timeline
                .gain_at_frame(fps, None)
                .expect("last frame should be inside segment");
            assert!(
                (g_end - (-20.0)).abs() < 0.5,
                "{fr:?}: gain at frame {fps} should be ≈ -20 dB, got {g_end}"
            );
        }
    }

    /// Verify that the `FrameRate::as_u8` values are correct for all variants.
    #[test]
    fn test_frame_rate_as_u8_all_variants() {
        assert_eq!(FrameRate::Fps24.as_u8(), 24);
        assert_eq!(FrameRate::Fps25.as_u8(), 25);
        assert_eq!(FrameRate::Fps2997Df.as_u8(), 30);
        assert_eq!(FrameRate::Fps2997Ndf.as_u8(), 30);
        assert_eq!(FrameRate::Fps30.as_u8(), 30);
        assert_eq!(FrameRate::Fps50.as_u8(), 50);
        assert_eq!(FrameRate::Fps5994.as_u8(), 60);
        assert_eq!(FrameRate::Fps60.as_u8(), 60);
    }

    /// Timecode ordering: a timecode at 01:00:00:00 must be greater than 00:59:59:23
    /// at 24fps, and equality must hold for identical timecodes.
    #[test]
    fn test_timecode_ordering_across_hour_boundary_24fps() {
        let just_before = Timecode::new(0, 59, 59, 23, FrameRate::Fps24);
        let one_hour = Timecode::new(1, 0, 0, 0, FrameRate::Fps24);
        let one_hour_2 = Timecode::new(1, 0, 0, 0, FrameRate::Fps24);

        assert!(
            one_hour > just_before,
            "01:00:00:00 must be greater than 00:59:59:23"
        );
        assert_eq!(one_hour, one_hour_2, "identical timecodes must be equal");
    }
}
