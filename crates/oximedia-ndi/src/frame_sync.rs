//! NDI frame synchronisation.
//!
//! Provides timestamp arithmetic and a dual-queue buffer that tracks video and audio
//! frame arrival times, enabling detection and correction of audio/video sync drift.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]

use std::collections::VecDeque;

/// A high-resolution timestamp using 100-nanosecond ticks, matching the NDI wire format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FrameTimestamp {
    /// Presentation timestamp in 100 ns units.
    pub pts_100ns: u64,
    /// Timecode in 100 ns units (may differ from PTS for drop-frame formats).
    pub timecode_100ns: u64,
}

impl FrameTimestamp {
    /// Construct a `FrameTimestamp` from a millisecond value.
    ///
    /// Both `pts_100ns` and `timecode_100ns` are set to the same converted value.
    pub fn from_ms(ms: u64) -> Self {
        let ticks = ms * 10_000; // 1 ms = 10 000 × 100 ns
        Self {
            pts_100ns: ticks,
            timecode_100ns: ticks,
        }
    }

    /// Convert the PTS back to milliseconds (rounding down).
    pub fn to_ms(&self) -> u64 {
        self.pts_100ns / 10_000
    }

    /// Signed difference in 100-ns ticks: `self.pts_100ns − other.pts_100ns`.
    ///
    /// Positive values mean `self` is later than `other`.
    pub fn diff(&self, other: &Self) -> i64 {
        self.pts_100ns as i64 - other.pts_100ns as i64
    }
}

/// Dual-queue buffer that collects timestamped video and audio frames and reports
/// whether the two streams are in sync.
#[derive(Debug)]
pub struct FrameSyncBuffer {
    /// Timestamps of received video frames (oldest first).
    pub video_frames: VecDeque<FrameTimestamp>,
    /// Timestamps of received audio frames (oldest first).
    pub audio_frames: VecDeque<FrameTimestamp>,
    /// Maximum tolerated absolute A/V drift in milliseconds before the streams
    /// are considered out of sync.
    pub max_drift_ms: u64,
}

impl FrameSyncBuffer {
    /// Create a new buffer with the given drift tolerance.
    pub fn new(max_drift_ms: u64) -> Self {
        Self {
            video_frames: VecDeque::new(),
            audio_frames: VecDeque::new(),
            max_drift_ms,
        }
    }

    /// Record the arrival of a video frame.
    pub fn push_video(&mut self, ts: FrameTimestamp) {
        self.video_frames.push_back(ts);
    }

    /// Record the arrival of an audio frame.
    pub fn push_audio(&mut self, ts: FrameTimestamp) {
        self.audio_frames.push_back(ts);
    }

    /// Return the current A/V sync error in milliseconds.
    ///
    /// The value is computed as `latest_video_pts − latest_audio_pts` (in ms).
    /// Positive means video is ahead of audio; negative means audio is ahead.
    /// Returns `None` if either queue is empty.
    pub fn av_sync_error_ms(&self) -> Option<f64> {
        let v = self.video_frames.back()?;
        let a = self.audio_frames.back()?;
        let diff_ticks = v.pts_100ns as i64 - a.pts_100ns as i64;
        Some(diff_ticks as f64 / 10_000.0)
    }

    /// Return `true` when both streams are present and the absolute A/V drift is
    /// within `max_drift_ms`.
    pub fn is_synchronized(&self) -> bool {
        match self.av_sync_error_ms() {
            Some(err) => err.abs() <= self.max_drift_ms as f64,
            None => false,
        }
    }

    /// Remove all frames whose PTS (converted to ms) is strictly less than `cutoff_ms`.
    pub fn drain_old_frames(&mut self, cutoff_ms: u64) {
        let cutoff_ticks = cutoff_ms * 10_000;
        while self
            .video_frames
            .front()
            .map_or(false, |f| f.pts_100ns < cutoff_ticks)
        {
            self.video_frames.pop_front();
        }
        while self
            .audio_frames
            .front()
            .map_or(false, |f| f.pts_100ns < cutoff_ticks)
        {
            self.audio_frames.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_ms_roundtrip() {
        let ts = FrameTimestamp::from_ms(1000);
        assert_eq!(ts.to_ms(), 1000);
    }

    #[test]
    fn test_from_ms_ticks() {
        let ts = FrameTimestamp::from_ms(1);
        assert_eq!(ts.pts_100ns, 10_000);
        assert_eq!(ts.timecode_100ns, 10_000);
    }

    #[test]
    fn test_diff_positive() {
        let a = FrameTimestamp::from_ms(100);
        let b = FrameTimestamp::from_ms(50);
        assert_eq!(a.diff(&b), 500_000); // 50 ms = 500 000 × 100 ns
    }

    #[test]
    fn test_diff_negative() {
        let a = FrameTimestamp::from_ms(50);
        let b = FrameTimestamp::from_ms(100);
        assert_eq!(a.diff(&b), -500_000);
    }

    #[test]
    fn test_diff_zero() {
        let a = FrameTimestamp::from_ms(200);
        assert_eq!(a.diff(&a), 0);
    }

    #[test]
    fn test_av_sync_error_none_when_empty() {
        let buf = FrameSyncBuffer::new(40);
        assert_eq!(buf.av_sync_error_ms(), None);
    }

    #[test]
    fn test_av_sync_error_some() {
        let mut buf = FrameSyncBuffer::new(40);
        buf.push_video(FrameTimestamp::from_ms(100));
        buf.push_audio(FrameTimestamp::from_ms(80));
        let err = buf.av_sync_error_ms().expect("expected sync error value");
        assert!((err - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_is_synchronized_within_tolerance() {
        let mut buf = FrameSyncBuffer::new(40);
        buf.push_video(FrameTimestamp::from_ms(1000));
        buf.push_audio(FrameTimestamp::from_ms(1010));
        assert!(buf.is_synchronized());
    }

    #[test]
    fn test_is_synchronized_exceeds_tolerance() {
        let mut buf = FrameSyncBuffer::new(10);
        buf.push_video(FrameTimestamp::from_ms(1000));
        buf.push_audio(FrameTimestamp::from_ms(1050));
        assert!(!buf.is_synchronized());
    }

    #[test]
    fn test_is_synchronized_false_when_empty() {
        let buf = FrameSyncBuffer::new(40);
        assert!(!buf.is_synchronized());
    }

    #[test]
    fn test_drain_old_frames() {
        let mut buf = FrameSyncBuffer::new(40);
        buf.push_video(FrameTimestamp::from_ms(10));
        buf.push_video(FrameTimestamp::from_ms(500));
        buf.push_audio(FrameTimestamp::from_ms(5));
        buf.push_audio(FrameTimestamp::from_ms(600));
        buf.drain_old_frames(100);
        assert_eq!(buf.video_frames.len(), 1);
        assert_eq!(
            buf.video_frames
                .front()
                .expect("expected non-empty deque")
                .to_ms(),
            500
        );
        assert_eq!(buf.audio_frames.len(), 1);
        assert_eq!(
            buf.audio_frames
                .front()
                .expect("expected non-empty deque")
                .to_ms(),
            600
        );
    }

    #[test]
    fn test_drain_removes_all_old() {
        let mut buf = FrameSyncBuffer::new(40);
        buf.push_video(FrameTimestamp::from_ms(1));
        buf.push_video(FrameTimestamp::from_ms(2));
        buf.drain_old_frames(1000);
        assert!(buf.video_frames.is_empty());
    }

    #[test]
    fn test_ordering_pts() {
        let a = FrameTimestamp::from_ms(100);
        let b = FrameTimestamp::from_ms(200);
        assert!(a < b);
    }
}
