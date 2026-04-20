//! ISO recording management for multicam production.
//!
//! Tracks per-angle ISO recording channels with state transitions and duration accounting.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Recording state of a single ISO channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsoRecordState {
    /// Channel is idle and not configured.
    Idle,
    /// Channel is armed and ready to record on command.
    Armed,
    /// Channel is actively recording.
    Recording,
    /// Recording has been stopped normally.
    Stopped,
    /// An error occurred during recording.
    Error,
}

impl IsoRecordState {
    /// Returns `true` when the channel is currently recording.
    #[must_use]
    pub fn is_active(&self) -> bool {
        *self == IsoRecordState::Recording
    }
}

/// A single ISO recording channel tied to one camera angle.
#[derive(Debug, Clone)]
pub struct IsoChannel {
    /// Angle this channel records.
    pub angle_id: u8,
    /// Current recording state.
    pub state: IsoRecordState,
    /// Number of frames recorded so far.
    pub duration_frames: u64,
    /// Output file path.
    pub file_path: String,
}

impl IsoChannel {
    /// Transitions the channel from `Armed` to `Recording`.
    ///
    /// No-ops if the channel is not `Armed`.
    pub fn start_recording(&mut self) {
        if self.state == IsoRecordState::Armed {
            self.state = IsoRecordState::Recording;
        }
    }

    /// Transitions the channel from `Recording` to `Stopped`.
    ///
    /// No-ops if the channel is not `Recording`.
    pub fn stop_recording(&mut self) {
        if self.state == IsoRecordState::Recording {
            self.state = IsoRecordState::Stopped;
        }
    }

    /// Returns the recorded duration in seconds at the given frame rate.
    #[must_use]
    pub fn duration_seconds(&self, fps: f32) -> f32 {
        if fps <= 0.0 {
            return 0.0;
        }
        self.duration_frames as f32 / fps
    }
}

/// Manages a collection of ISO recording channels.
#[derive(Debug, Clone, Default)]
pub struct IsoRecordManager {
    /// All registered channels.
    pub channels: Vec<IsoChannel>,
}

impl IsoRecordManager {
    /// Registers a new channel for `angle_id` with the given output path.
    pub fn add_channel(&mut self, angle_id: u8, path: impl Into<String>) {
        self.channels.push(IsoChannel {
            angle_id,
            state: IsoRecordState::Idle,
            duration_frames: 0,
            file_path: path.into(),
        });
    }

    /// Arms all channels that are currently `Idle`.
    pub fn arm_all(&mut self) {
        for ch in &mut self.channels {
            if ch.state == IsoRecordState::Idle {
                ch.state = IsoRecordState::Armed;
            }
        }
    }

    /// Starts recording on all `Armed` channels.
    pub fn start_all(&mut self) {
        for ch in &mut self.channels {
            ch.start_recording();
        }
    }

    /// Stops recording on all `Recording` channels.
    pub fn stop_all(&mut self) {
        for ch in &mut self.channels {
            ch.stop_recording();
        }
    }

    /// Returns references to all channels that are actively recording.
    #[must_use]
    pub fn recording_channels(&self) -> Vec<&IsoChannel> {
        self.channels
            .iter()
            .filter(|ch| ch.state.is_active())
            .collect()
    }

    /// Returns the sum of `duration_frames` across all channels.
    #[must_use]
    pub fn total_duration_frames(&self) -> u64 {
        self.channels.iter().map(|ch| ch.duration_frames).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- IsoRecordState ---

    #[test]
    fn test_recording_is_active() {
        assert!(IsoRecordState::Recording.is_active());
    }

    #[test]
    fn test_idle_not_active() {
        assert!(!IsoRecordState::Idle.is_active());
    }

    #[test]
    fn test_armed_not_active() {
        assert!(!IsoRecordState::Armed.is_active());
    }

    #[test]
    fn test_stopped_not_active() {
        assert!(!IsoRecordState::Stopped.is_active());
    }

    #[test]
    fn test_error_not_active() {
        assert!(!IsoRecordState::Error.is_active());
    }

    // --- IsoChannel ---

    fn tmp_str(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("oximedia-multicam-iso-{name}"))
            .to_string_lossy()
            .into_owned()
    }

    fn make_channel(state: IsoRecordState) -> IsoChannel {
        IsoChannel {
            angle_id: 1,
            state,
            duration_frames: 300,
            file_path: tmp_str("cam1.mov"),
        }
    }

    #[test]
    fn test_start_recording_from_armed() {
        let mut ch = make_channel(IsoRecordState::Armed);
        ch.start_recording();
        assert_eq!(ch.state, IsoRecordState::Recording);
    }

    #[test]
    fn test_start_recording_noop_from_idle() {
        let mut ch = make_channel(IsoRecordState::Idle);
        ch.start_recording();
        assert_eq!(ch.state, IsoRecordState::Idle);
    }

    #[test]
    fn test_stop_recording_from_recording() {
        let mut ch = make_channel(IsoRecordState::Recording);
        ch.stop_recording();
        assert_eq!(ch.state, IsoRecordState::Stopped);
    }

    #[test]
    fn test_stop_recording_noop_from_armed() {
        let mut ch = make_channel(IsoRecordState::Armed);
        ch.stop_recording();
        assert_eq!(ch.state, IsoRecordState::Armed);
    }

    #[test]
    fn test_duration_seconds() {
        let ch = make_channel(IsoRecordState::Stopped); // 300 frames
        assert!((ch.duration_seconds(25.0) - 12.0).abs() < 0.001);
    }

    #[test]
    fn test_duration_seconds_zero_fps() {
        let ch = make_channel(IsoRecordState::Stopped);
        assert_eq!(ch.duration_seconds(0.0), 0.0);
    }

    // --- IsoRecordManager ---

    fn make_manager() -> IsoRecordManager {
        let mut mgr = IsoRecordManager::default();
        mgr.add_channel(0, &tmp_str("cam0.mov"));
        mgr.add_channel(1, &tmp_str("mgr_cam1.mov"));
        mgr
    }

    #[test]
    fn test_add_channel() {
        let mgr = make_manager();
        assert_eq!(mgr.channels.len(), 2);
    }

    #[test]
    fn test_arm_all() {
        let mut mgr = make_manager();
        mgr.arm_all();
        assert!(mgr
            .channels
            .iter()
            .all(|ch| ch.state == IsoRecordState::Armed));
    }

    #[test]
    fn test_start_all() {
        let mut mgr = make_manager();
        mgr.arm_all();
        mgr.start_all();
        assert!(mgr
            .channels
            .iter()
            .all(|ch| ch.state == IsoRecordState::Recording));
    }

    #[test]
    fn test_stop_all() {
        let mut mgr = make_manager();
        mgr.arm_all();
        mgr.start_all();
        mgr.stop_all();
        assert!(mgr
            .channels
            .iter()
            .all(|ch| ch.state == IsoRecordState::Stopped));
    }

    #[test]
    fn test_recording_channels_count() {
        let mut mgr = make_manager();
        mgr.arm_all();
        mgr.start_all();
        assert_eq!(mgr.recording_channels().len(), 2);
    }

    #[test]
    fn test_recording_channels_empty_when_stopped() {
        let mut mgr = make_manager();
        mgr.arm_all();
        mgr.start_all();
        mgr.stop_all();
        assert!(mgr.recording_channels().is_empty());
    }

    #[test]
    fn test_total_duration_frames() {
        let mut mgr = make_manager();
        mgr.channels[0].duration_frames = 100;
        mgr.channels[1].duration_frames = 200;
        assert_eq!(mgr.total_duration_frames(), 300);
    }
}
