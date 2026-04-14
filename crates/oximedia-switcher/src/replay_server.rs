//! Instant replay server for live production switchers.
//!
//! Maintains a circular ring-buffer of captured frame metadata (by timestamp)
//! and provides variable-speed playback control (slow-motion, normal, and
//! fast-forward).  Actual pixel data management is left to a downstream
//! frame-store; this module tracks *which* frames to play and at what speed.
//!
//! # Concepts
//!
//! - **Clip** — a contiguous range of frames selected from the ring buffer.
//! - **Playback channel** — an independent playback head that can be assigned
//!   a clip and played at any speed.
//! - **Ring buffer** — a fixed-capacity FIFO of incoming frame references
//!   (frame numbers / timestamps).  When the buffer is full the oldest entry
//!   is silently discarded.
//! - **Speed** — a `f32` multiplier; `1.0` = real-time, `0.25` = quarter-speed
//!   slow-mo, `4.0` = 4× fast-forward.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::VecDeque;
use thiserror::Error;

/// Errors produced by the replay server.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ReplayError {
    /// The ring buffer contains no frames.
    #[error("replay buffer is empty")]
    BufferEmpty,
    /// The frame number is outside the buffered range.
    #[error("frame {0} is not in the replay buffer")]
    FrameNotBuffered(u64),
    /// The clip start exceeds the clip end.
    #[error("clip start {0} must be <= clip end {1}")]
    InvalidClipRange(u64, u64),
    /// The requested playback channel does not exist.
    #[error("playback channel {0} not found")]
    ChannelNotFound(usize),
    /// A playback channel with this ID is already registered.
    #[error("playback channel {0} already exists")]
    ChannelAlreadyExists(usize),
    /// The playback speed is out of the acceptable range.
    #[error("speed {0} is outside the valid range")]
    InvalidSpeed(f32),
    /// No clip is loaded in the channel.
    #[error("no clip loaded in channel {0}")]
    NoClipLoaded(usize),
    /// The replay server has reached maximum clip capacity.
    #[error("clip capacity ({0}) exceeded")]
    ClipCapacityExceeded(usize),
}

// ── Frame reference ───────────────────────────────────────────────────────────

/// A lightweight reference to a single captured frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FrameRef {
    /// Monotonically increasing frame number (from capture start).
    pub frame_number: u64,
    /// Presentation timestamp in microseconds.
    pub pts_us: u64,
}

impl FrameRef {
    /// Create a new frame reference.
    pub fn new(frame_number: u64, pts_us: u64) -> Self {
        Self {
            frame_number,
            pts_us,
        }
    }
}

// ── Replay clip ───────────────────────────────────────────────────────────────

/// A user-defined clip: a labelled range within the ring buffer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayClip {
    /// Unique clip identifier.
    pub id: usize,
    /// Human-readable label.
    pub label: String,
    /// First frame number (inclusive).
    pub start_frame: u64,
    /// Last frame number (inclusive).
    pub end_frame: u64,
    /// Default playback speed for this clip.
    pub default_speed: f32,
}

impl ReplayClip {
    /// Create a clip spanning `[start_frame, end_frame]`.
    pub fn new(
        id: usize,
        label: impl Into<String>,
        start_frame: u64,
        end_frame: u64,
    ) -> Result<Self, ReplayError> {
        if start_frame > end_frame {
            return Err(ReplayError::InvalidClipRange(start_frame, end_frame));
        }
        Ok(Self {
            id,
            label: label.into(),
            start_frame,
            end_frame,
            default_speed: 1.0,
        })
    }

    /// Total frame count in this clip.
    pub fn frame_count(&self) -> u64 {
        self.end_frame - self.start_frame + 1
    }
}

// ── Playback state ────────────────────────────────────────────────────────────

/// Playback direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlaybackDirection {
    /// Forward (normal or fast-forward).
    Forward,
    /// Reverse (slow-mo or rewind).
    Reverse,
}

/// Playback state of a single channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlaybackState {
    /// Channel is idle; no clip loaded.
    Idle,
    /// Actively playing.
    Playing,
    /// Playback is paused at the current position.
    Paused,
    /// Clip has played to completion.
    Finished,
}

/// A playback channel capable of playing a single clip at variable speed.
#[derive(Debug)]
pub struct PlaybackChannel {
    /// Channel ID.
    pub id: usize,
    /// Label for UI display.
    pub label: String,
    /// Currently loaded clip, if any.
    pub clip: Option<ReplayClip>,
    /// Playback state.
    pub state: PlaybackState,
    /// Playback speed (positive = forward, negative implies reverse direction).
    pub speed: f32,
    /// Direction of playback.
    pub direction: PlaybackDirection,
    /// Current fractional position within the clip (0.0 = start, 1.0 = end).
    current_position: f64,
    /// Accumulated sub-frame residual for fractional speed playback.
    residual: f64,
}

impl PlaybackChannel {
    /// Create a new idle playback channel.
    pub fn new(id: usize, label: impl Into<String>) -> Self {
        Self {
            id,
            label: label.into(),
            clip: None,
            state: PlaybackState::Idle,
            speed: 1.0,
            direction: PlaybackDirection::Forward,
            current_position: 0.0,
            residual: 0.0,
        }
    }

    /// Load a clip into this channel, resetting position to the beginning.
    pub fn load_clip(&mut self, clip: ReplayClip) {
        self.clip = Some(clip);
        self.state = PlaybackState::Paused;
        self.current_position = 0.0;
        self.residual = 0.0;
    }

    /// Unload any loaded clip.
    pub fn unload_clip(&mut self) {
        self.clip = None;
        self.state = PlaybackState::Idle;
        self.current_position = 0.0;
        self.residual = 0.0;
    }

    /// Set the playback speed.  Must be in `[0.01, 32.0]`.
    pub fn set_speed(&mut self, speed: f32) -> Result<(), ReplayError> {
        if speed < 0.01 || speed > 32.0 {
            return Err(ReplayError::InvalidSpeed(speed));
        }
        self.speed = speed;
        Ok(())
    }

    /// Begin or resume playback.
    pub fn play(&mut self) -> Result<(), ReplayError> {
        if self.clip.is_none() {
            return Err(ReplayError::NoClipLoaded(self.id));
        }
        self.state = PlaybackState::Playing;
        Ok(())
    }

    /// Pause playback at the current position.
    pub fn pause(&mut self) {
        if self.state == PlaybackState::Playing {
            self.state = PlaybackState::Paused;
        }
    }

    /// Jump to the beginning of the loaded clip.
    pub fn rewind(&mut self) {
        self.current_position = 0.0;
        self.residual = 0.0;
        if self.state == PlaybackState::Finished {
            self.state = PlaybackState::Paused;
        }
    }

    /// Jump to the given fractional position `[0.0, 1.0]`.
    pub fn seek(&mut self, position: f64) -> Result<(), ReplayError> {
        if self.clip.is_none() {
            return Err(ReplayError::NoClipLoaded(self.id));
        }
        self.current_position = position.clamp(0.0, 1.0);
        self.residual = 0.0;
        Ok(())
    }

    /// Advance the playback head by one render frame.
    ///
    /// The channel accumulates fractional progress according to `speed` and
    /// the number of frames in the loaded clip.  Returns the `FrameRef`
    /// that should be displayed for this render frame, or `None` if the
    /// channel is not playing or has no clip.
    ///
    /// `available` is the ring buffer's ordered frame list (used to resolve
    /// the current position to a concrete `FrameRef`).
    pub fn advance(&mut self, available: &[FrameRef]) -> Option<FrameRef> {
        if self.state != PlaybackState::Playing {
            return self.current_frame_ref(available);
        }

        let clip = self.clip.as_ref()?;
        let frame_count = clip.frame_count() as f64;
        if frame_count == 0.0 {
            return None;
        }

        let step = self.speed as f64 / frame_count;
        match self.direction {
            PlaybackDirection::Forward => {
                self.residual += step;
                while self.residual >= 1.0 / frame_count {
                    self.current_position += 1.0 / frame_count;
                    self.residual -= 1.0 / frame_count;
                }
                if self.current_position >= 1.0 {
                    self.current_position = 1.0;
                    self.state = PlaybackState::Finished;
                }
            }
            PlaybackDirection::Reverse => {
                self.residual += step;
                while self.residual >= 1.0 / frame_count {
                    self.current_position -= 1.0 / frame_count;
                    self.residual -= 1.0 / frame_count;
                }
                if self.current_position <= 0.0 {
                    self.current_position = 0.0;
                    self.state = PlaybackState::Finished;
                }
            }
        }

        self.current_frame_ref(available)
    }

    /// Resolve the current fractional position to a concrete `FrameRef`.
    fn current_frame_ref(&self, available: &[FrameRef]) -> Option<FrameRef> {
        let clip = self.clip.as_ref()?;
        // Find the frames within the clip's range from the buffer.
        let clip_frames: Vec<&FrameRef> = available
            .iter()
            .filter(|f| f.frame_number >= clip.start_frame && f.frame_number <= clip.end_frame)
            .collect();
        if clip_frames.is_empty() {
            return None;
        }
        let idx = ((self.current_position * (clip_frames.len() as f64 - f64::EPSILON)).floor()
            as usize)
            .min(clip_frames.len() - 1);
        Some(*clip_frames[idx])
    }

    /// Current fractional position within the loaded clip.
    pub fn position(&self) -> f64 {
        self.current_position
    }
}

// ── Ring buffer ───────────────────────────────────────────────────────────────

/// Fixed-capacity ring buffer for incoming frame references.
pub struct ReplayRingBuffer {
    buffer: VecDeque<FrameRef>,
    capacity: usize,
}

impl ReplayRingBuffer {
    /// Create a ring buffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Push a frame reference into the buffer.  If full, the oldest frame
    /// is discarded.
    pub fn push(&mut self, frame: FrameRef) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(frame);
    }

    /// Number of buffered frames.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns `true` if the buffer contains no frames.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Returns the oldest buffered frame reference.
    pub fn oldest(&self) -> Option<FrameRef> {
        self.buffer.front().copied()
    }

    /// Returns the newest buffered frame reference.
    pub fn newest(&self) -> Option<FrameRef> {
        self.buffer.back().copied()
    }

    /// Returns `true` if the given frame number is present in the buffer.
    pub fn contains(&self, frame_number: u64) -> bool {
        self.buffer.iter().any(|f| f.frame_number == frame_number)
    }

    /// Return a slice-like view (sorted ascending by frame number) for clip
    /// resolution.
    pub fn frames_sorted(&self) -> Vec<FrameRef> {
        let mut v: Vec<FrameRef> = self.buffer.iter().copied().collect();
        v.sort_by_key(|f| f.frame_number);
        v
    }

    /// Create a clip spanning the most recent `frame_count` frames.
    pub fn clip_recent(
        &self,
        id: usize,
        label: impl Into<String>,
        frame_count: u64,
    ) -> Result<ReplayClip, ReplayError> {
        let newest = self.newest().ok_or(ReplayError::BufferEmpty)?;
        let oldest = self.oldest().ok_or(ReplayError::BufferEmpty)?;
        let start = newest
            .frame_number
            .saturating_sub(frame_count.saturating_sub(1));
        let start = start.max(oldest.frame_number);
        ReplayClip::new(id, label, start, newest.frame_number)
    }
}

// ── Replay server ─────────────────────────────────────────────────────────────

/// Central replay server: one ring buffer, multiple playback channels,
/// and a library of user-defined clips.
pub struct ReplayServer {
    /// The frame ring buffer.
    pub ring: ReplayRingBuffer,
    /// Saved replay clips.
    clips: HashMap<usize, ReplayClip>,
    /// Playback channels.
    channels: HashMap<usize, PlaybackChannel>,
    /// Maximum number of saved clips.
    clip_capacity: usize,
}

impl ReplayServer {
    /// Create a new replay server.
    ///
    /// `buffer_capacity` — number of frames kept in the live ring buffer.
    /// `clip_capacity`   — maximum number of saved clips.
    pub fn new(buffer_capacity: usize, clip_capacity: usize) -> Self {
        Self {
            ring: ReplayRingBuffer::new(buffer_capacity),
            clips: HashMap::new(),
            channels: HashMap::new(),
            clip_capacity,
        }
    }

    // ── Frame ingestion ───────────────────────────────────────────────────────

    /// Push a frame into the ring buffer (called every frame during live capture).
    pub fn ingest_frame(&mut self, frame: FrameRef) {
        self.ring.push(frame);
    }

    // ── Clip management ───────────────────────────────────────────────────────

    /// Save a clip by ID.
    pub fn save_clip(&mut self, clip: ReplayClip) -> Result<(), ReplayError> {
        if !self.clips.contains_key(&clip.id) && self.clips.len() >= self.clip_capacity {
            return Err(ReplayError::ClipCapacityExceeded(self.clip_capacity));
        }
        self.clips.insert(clip.id, clip);
        Ok(())
    }

    /// Delete a saved clip.
    pub fn delete_clip(&mut self, id: usize) -> Result<(), ReplayError> {
        self.clips
            .remove(&id)
            .ok_or(ReplayError::ChannelNotFound(id))?;
        Ok(())
    }

    /// Get a saved clip.
    pub fn clip(&self, id: usize) -> Option<&ReplayClip> {
        self.clips.get(&id)
    }

    /// Number of saved clips.
    pub fn clip_count(&self) -> usize {
        self.clips.len()
    }

    /// Create a clip spanning the most recent `frame_count` frames and save it.
    pub fn mark_recent(
        &mut self,
        id: usize,
        label: impl Into<String>,
        frame_count: u64,
    ) -> Result<(), ReplayError> {
        let clip = self.ring.clip_recent(id, label, frame_count)?;
        self.save_clip(clip)
    }

    // ── Channel management ────────────────────────────────────────────────────

    /// Register a new playback channel.
    pub fn add_channel(&mut self, channel: PlaybackChannel) -> Result<(), ReplayError> {
        if self.channels.contains_key(&channel.id) {
            return Err(ReplayError::ChannelAlreadyExists(channel.id));
        }
        self.channels.insert(channel.id, channel);
        Ok(())
    }

    /// Remove a playback channel.
    pub fn remove_channel(&mut self, id: usize) -> Result<(), ReplayError> {
        self.channels
            .remove(&id)
            .ok_or(ReplayError::ChannelNotFound(id))?;
        Ok(())
    }

    /// Get a reference to a playback channel.
    pub fn channel(&self, id: usize) -> Option<&PlaybackChannel> {
        self.channels.get(&id)
    }

    /// Get a mutable reference to a playback channel.
    pub fn channel_mut(&mut self, id: usize) -> Option<&mut PlaybackChannel> {
        self.channels.get_mut(&id)
    }

    /// Number of registered channels.
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    // ── Convenience operations ────────────────────────────────────────────────

    /// Load a saved clip into a channel and begin playing at the given speed.
    pub fn play_clip(
        &mut self,
        channel_id: usize,
        clip_id: usize,
        speed: f32,
    ) -> Result<(), ReplayError> {
        let clip = self
            .clips
            .get(&clip_id)
            .ok_or(ReplayError::ChannelNotFound(clip_id))?
            .clone();
        let channel = self
            .channels
            .get_mut(&channel_id)
            .ok_or(ReplayError::ChannelNotFound(channel_id))?;
        channel.set_speed(speed)?;
        channel.load_clip(clip);
        channel.play()
    }

    /// Advance all playing channels by one render frame.
    ///
    /// Returns a map of `channel_id → FrameRef` for all channels that yielded
    /// a frame this tick.
    pub fn tick(&mut self) -> HashMap<usize, FrameRef> {
        let frames = self.ring.frames_sorted();
        let mut output = HashMap::new();
        for (id, channel) in &mut self.channels {
            if let Some(frame_ref) = channel.advance(&frames) {
                output.insert(*id, frame_ref);
            }
        }
        output
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn populated_ring(n: u64) -> ReplayRingBuffer {
        let mut ring = ReplayRingBuffer::new(512);
        for i in 0..n {
            ring.push(FrameRef::new(i, i * 40_000)); // 25 fps pts
        }
        ring
    }

    fn server_with_frames(n: u64) -> ReplayServer {
        let mut srv = ReplayServer::new(512, 64);
        for i in 0..n {
            srv.ingest_frame(FrameRef::new(i, i * 40_000));
        }
        srv
    }

    #[test]
    fn test_ring_buffer_push_and_len() {
        let ring = populated_ring(50);
        assert_eq!(ring.len(), 50);
    }

    #[test]
    fn test_ring_buffer_overflow_discards_oldest() {
        let mut ring = ReplayRingBuffer::new(4);
        for i in 0..6u64 {
            ring.push(FrameRef::new(i, i * 1000));
        }
        assert_eq!(ring.len(), 4);
        assert_eq!(ring.oldest().expect("oldest").frame_number, 2);
        assert_eq!(ring.newest().expect("newest").frame_number, 5);
    }

    #[test]
    fn test_ring_buffer_contains() {
        let ring = populated_ring(10);
        assert!(ring.contains(5));
        assert!(!ring.contains(99));
    }

    #[test]
    fn test_clip_recent_basic() {
        let ring = populated_ring(100);
        let clip = ring.clip_recent(1, "Last 25", 25).expect("clip");
        assert_eq!(clip.frame_count(), 25);
        assert_eq!(clip.end_frame, 99);
    }

    #[test]
    fn test_clip_invalid_range_errors() {
        let err = ReplayClip::new(0, "bad", 10, 5);
        assert_eq!(err, Err(ReplayError::InvalidClipRange(10, 5)));
    }

    #[test]
    fn test_playback_channel_load_and_play() {
        let clip = ReplayClip::new(0, "Test", 0, 49).expect("clip");
        let mut ch = PlaybackChannel::new(0, "Ch 1");
        assert_eq!(ch.state, PlaybackState::Idle);
        ch.load_clip(clip);
        assert_eq!(ch.state, PlaybackState::Paused);
        ch.play().expect("play");
        assert_eq!(ch.state, PlaybackState::Playing);
    }

    #[test]
    fn test_playback_channel_invalid_speed_errors() {
        let mut ch = PlaybackChannel::new(0, "Ch");
        assert!(matches!(
            ch.set_speed(0.0),
            Err(ReplayError::InvalidSpeed(_))
        ));
        assert!(matches!(
            ch.set_speed(100.0),
            Err(ReplayError::InvalidSpeed(_))
        ));
    }

    #[test]
    fn test_playback_channel_advance_returns_frame() {
        let frames: Vec<FrameRef> = (0..50u64).map(|i| FrameRef::new(i, i * 40_000)).collect();
        let clip = ReplayClip::new(0, "C", 0, 49).expect("clip");
        let mut ch = PlaybackChannel::new(0, "Ch");
        ch.load_clip(clip);
        ch.play().expect("play");
        let result = ch.advance(&frames);
        assert!(result.is_some());
    }

    #[test]
    fn test_playback_channel_slow_mo_advance() {
        let frames: Vec<FrameRef> = (0..50u64).map(|i| FrameRef::new(i, i * 40_000)).collect();
        let clip = ReplayClip::new(0, "SM", 0, 49).expect("clip");
        let mut ch = PlaybackChannel::new(0, "Ch");
        ch.set_speed(0.25).expect("speed");
        ch.load_clip(clip);
        ch.play().expect("play");
        let pos_before = ch.position();
        ch.advance(&frames);
        let pos_after = ch.position();
        assert!(pos_after >= pos_before, "position should advance or stay");
    }

    #[test]
    fn test_server_play_clip() {
        let mut srv = server_with_frames(100);
        srv.mark_recent(1, "Last 25", 25).expect("mark");
        let ch = PlaybackChannel::new(0, "Chan 0");
        srv.add_channel(ch).expect("add");
        srv.play_clip(0, 1, 0.5).expect("play");
        assert_eq!(srv.channel(0).expect("ch").state, PlaybackState::Playing);
    }

    #[test]
    fn test_server_tick_returns_frame_refs() {
        let mut srv = server_with_frames(60);
        srv.mark_recent(1, "Clip", 30).expect("mark");
        let ch = PlaybackChannel::new(0, "Ch");
        srv.add_channel(ch).expect("add");
        srv.play_clip(0, 1, 1.0).expect("play");
        let output = srv.tick();
        assert!(output.contains_key(&0));
    }

    #[test]
    fn test_server_add_duplicate_channel_errors() {
        let mut srv = server_with_frames(10);
        srv.add_channel(PlaybackChannel::new(1, "A")).expect("ok");
        let err = srv.add_channel(PlaybackChannel::new(1, "B"));
        assert_eq!(err, Err(ReplayError::ChannelAlreadyExists(1)));
    }

    #[test]
    fn test_server_clip_capacity_exceeded() {
        let mut srv = ReplayServer::new(512, 2);
        for i in 0..5u64 {
            srv.ingest_frame(FrameRef::new(i, i * 1000));
        }
        srv.mark_recent(1, "C1", 2).expect("c1");
        srv.mark_recent(2, "C2", 2).expect("c2");
        let err = srv.mark_recent(3, "C3", 2);
        assert_eq!(err, Err(ReplayError::ClipCapacityExceeded(2)));
    }

    #[test]
    fn test_rewind_resets_position() {
        let mut ch = PlaybackChannel::new(0, "Ch");
        let clip = ReplayClip::new(0, "R", 0, 9).expect("clip");
        ch.load_clip(clip);
        ch.seek(0.8).expect("seek");
        assert!((ch.position() - 0.8).abs() < 0.01);
        ch.rewind();
        assert_eq!(ch.position(), 0.0);
    }

    #[test]
    fn test_frame_ref_ordering() {
        let a = FrameRef::new(10, 400_000);
        let b = FrameRef::new(20, 800_000);
        assert!(a < b);
    }

    #[test]
    fn test_ring_buffer_empty() {
        let ring = ReplayRingBuffer::new(100);
        assert!(ring.is_empty());
        assert!(ring.oldest().is_none());
        assert!(ring.newest().is_none());
    }
}
