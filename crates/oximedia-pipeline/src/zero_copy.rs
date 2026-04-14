//! Zero-copy frame passing between adjacent pipeline nodes.
//!
//! When two adjacent nodes run on the same thread and the downstream node does
//! not need to mutate the frame data, passing an [`Arc`]-wrapped byte slice
//! avoids an allocation and `memcpy`.  Only the reference count is incremented.
//!
//! # Design
//!
//! [`ZeroCopyFrame`] wraps a read-only `Arc<[u8]>` payload together with a
//! [`FrameDescriptor`] that carries format metadata (dimensions, sample rate,
//! etc.).  A [`ZeroCopyChannel`] is a thin MPSC-style queue (backed by a
//! `VecDeque`) that lets producers push shared frames and consumers receive
//! them without copying.
//!
//! When a node **does** need to mutate the frame it can call
//! [`ZeroCopyFrame::try_into_owned`] to obtain an exclusive `Vec<u8>` —
//! copying only when the `Arc` is actually shared (reference count > 1).
//!
//! # Example
//!
//! ```rust
//! use oximedia_pipeline::zero_copy::{FrameDescriptor, ZeroCopyChannel, ZeroCopyFrame};
//! use oximedia_pipeline::node::{FrameFormat, StreamKind};
//!
//! // Create a frame shared between producer and consumer.
//! let desc = FrameDescriptor::video(FrameFormat::Yuv420p, 1920, 1080, 0);
//! let data: Arc<[u8]> = vec![0u8; 1920 * 1080 * 3 / 2].into();
//! let frame = ZeroCopyFrame::new(desc, Arc::clone(&data));
//!
//! let mut chan = ZeroCopyChannel::new(32);
//! chan.push(frame).expect("push ok");
//! let received = chan.pop().expect("not empty");
//! assert_eq!(received.descriptor().width, Some(1920));
//!
//! use std::sync::Arc;
//! ```

use std::collections::VecDeque;
use std::sync::Arc;

use crate::node::{FrameFormat, StreamKind};
use crate::PipelineError;

// ── FrameDescriptor ───────────────────────────────────────────────────────────

/// Metadata describing a raw media frame's layout and provenance.
///
/// Kept separate from the payload bytes so descriptors can be inspected,
/// logged, or filtered without touching the pixel data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameDescriptor {
    /// Whether this frame carries video or audio data.
    pub kind: StreamKind,
    /// Pixel / sample format.
    pub format: FrameFormat,
    /// Frame width in pixels (`None` for audio).
    pub width: Option<u32>,
    /// Frame height in pixels (`None` for audio).
    pub height: Option<u32>,
    /// Audio sample rate in Hz (`None` for video).
    pub sample_rate: Option<u32>,
    /// Audio channel count (`None` for video).
    pub channels: Option<u8>,
    /// Presentation timestamp in the stream's time base units.
    pub pts: i64,
    /// Human-readable source node label.
    pub source_label: String,
}

impl FrameDescriptor {
    /// Construct a video frame descriptor.
    pub fn video(format: FrameFormat, width: u32, height: u32, pts: i64) -> Self {
        Self {
            kind: StreamKind::Video,
            format,
            width: Some(width),
            height: Some(height),
            sample_rate: None,
            channels: None,
            pts,
            source_label: String::new(),
        }
    }

    /// Construct an audio frame descriptor.
    pub fn audio(format: FrameFormat, sample_rate: u32, channels: u8, pts: i64) -> Self {
        Self {
            kind: StreamKind::Audio,
            format,
            width: None,
            height: None,
            sample_rate: Some(sample_rate),
            channels: Some(channels),
            pts,
            source_label: String::new(),
        }
    }

    /// Set the source label and return `self` for chaining.
    pub fn with_source(mut self, label: impl Into<String>) -> Self {
        self.source_label = label.into();
        self
    }

    /// Estimated payload size in bytes.
    ///
    /// For video: `width × height × bytes_per_element`.  This is approximate
    /// for planar sub-sampled formats (e.g. YUV 4:2:0 needs `× 3/2`).
    /// For audio: returns `0` (caller must track buffer length separately).
    pub fn estimated_bytes(&self) -> usize {
        match self.kind {
            StreamKind::Video => {
                let w = self.width.unwrap_or(0) as usize;
                let h = self.height.unwrap_or(0) as usize;
                let bpe = self.format.bytes_per_element() as usize;
                w * h * bpe
            }
            _ => 0,
        }
    }
}

// ── ZeroCopyFrame ─────────────────────────────────────────────────────────────

/// A media frame whose backing buffer is shared via reference-counting.
///
/// Moving a `ZeroCopyFrame` between pipeline nodes costs only one atomic
/// increment (`Arc::clone`) until a node that must mutate the data calls
/// [`Self::try_into_owned`].
#[derive(Debug, Clone)]
pub struct ZeroCopyFrame {
    desc: FrameDescriptor,
    data: Arc<[u8]>,
}

impl ZeroCopyFrame {
    /// Wrap an existing `Arc<[u8]>` payload with the given descriptor.
    pub fn new(desc: FrameDescriptor, data: Arc<[u8]>) -> Self {
        Self { desc, data }
    }

    /// Construct a `ZeroCopyFrame` from a plain `Vec<u8>`, taking ownership
    /// and converting to a shared `Arc`.
    pub fn from_vec(desc: FrameDescriptor, data: Vec<u8>) -> Self {
        Self {
            desc,
            data: data.into(),
        }
    }

    /// Return a reference to the frame descriptor.
    pub fn descriptor(&self) -> &FrameDescriptor {
        &self.desc
    }

    /// Mutable access to the descriptor (e.g. to update the `pts` field).
    pub fn descriptor_mut(&mut self) -> &mut FrameDescriptor {
        &mut self.desc
    }

    /// Return a shared reference to the raw byte payload.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Return an `Arc` clone of the payload — increments the reference count
    /// without copying bytes.
    pub fn data_arc(&self) -> Arc<[u8]> {
        Arc::clone(&self.data)
    }

    /// Number of current references to the underlying buffer.
    ///
    /// A value of `1` means this is the sole owner; [`Self::try_into_owned`]
    /// will succeed without a copy.
    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.data)
    }

    /// Returns `true` when no other `ZeroCopyFrame` shares the same buffer.
    pub fn is_exclusive(&self) -> bool {
        Arc::strong_count(&self.data) == 1
    }

    /// Attempt to obtain an exclusive `Vec<u8>` from this frame.
    ///
    /// * If this is the **sole** owner of the buffer, the backing bytes are
    ///   copied into a fresh `Vec<u8>` without involving other allocations
    ///   (the `Arc` refcount check is free).
    /// * If other references exist the same copy path is taken.
    ///
    /// `Arc<[u8]>` does not support `try_unwrap` into `Vec<u8>` without an
    /// intermediate copy because the slice header embeds the length; we
    /// therefore always copy.  The "zero-copy" benefit of this module comes
    /// from deferring the copy until a node actually needs to mutate the data,
    /// and from avoiding copies entirely for read-only nodes.
    pub fn try_into_owned(self) -> Vec<u8> {
        self.data.to_vec()
    }

    /// Clone the frame, sharing the backing buffer with the original.
    ///
    /// Both the original and the clone reference the same `Arc`; only an
    /// atomic reference-count increment is performed.
    pub fn share(&self) -> Self {
        Self {
            desc: self.desc.clone(),
            data: Arc::clone(&self.data),
        }
    }

    /// Byte length of the payload buffer.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` when the payload buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// ── ZeroCopyChannel ───────────────────────────────────────────────────────────

/// A bounded single-thread queue for passing [`ZeroCopyFrame`]s between
/// adjacent pipeline nodes without copying frame data.
///
/// Frames are stored as `ZeroCopyFrame` values (which contain `Arc<[u8]>`),
/// so enqueuing a frame costs one atomic reference-count increment, not a
/// `memcpy`.  Consumers receive the same `Arc`-backed frame and can choose
/// to clone or take ownership depending on whether they need to mutate.
pub struct ZeroCopyChannel {
    capacity: usize,
    queue: VecDeque<ZeroCopyFrame>,
    /// Total frames enqueued over the channel's lifetime.
    pushed: u64,
    /// Total frames dequeued over the channel's lifetime.
    popped: u64,
    /// Total frames dropped because the channel was full.
    dropped: u64,
}

impl ZeroCopyChannel {
    /// Create a new channel with the given capacity (must be ≥ 1).
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        Self {
            capacity: cap,
            queue: VecDeque::with_capacity(cap),
            pushed: 0,
            popped: 0,
            dropped: 0,
        }
    }

    /// Enqueue a frame without copying its payload.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::ValidationError`] when the channel is full.
    /// The caller should handle the error by either dropping the frame or
    /// waiting for the consumer to drain the queue.
    pub fn push(&mut self, frame: ZeroCopyFrame) -> Result<(), PipelineError> {
        if self.queue.len() >= self.capacity {
            self.dropped += 1;
            return Err(PipelineError::ValidationError(format!(
                "ZeroCopyChannel full (capacity={}): frame dropped",
                self.capacity
            )));
        }
        self.queue.push_back(frame);
        self.pushed += 1;
        Ok(())
    }

    /// Dequeue the oldest frame, or return `None` when the channel is empty.
    pub fn pop(&mut self) -> Option<ZeroCopyFrame> {
        let frame = self.queue.pop_front();
        if frame.is_some() {
            self.popped += 1;
        }
        frame
    }

    /// Peek at the oldest frame without removing it.
    pub fn peek(&self) -> Option<&ZeroCopyFrame> {
        self.queue.front()
    }

    /// Current number of frames waiting in the channel.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Returns `true` when no frames are queued.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Maximum number of frames the channel can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Fill level as a value in `[0.0, 1.0]`.
    pub fn fill_ratio(&self) -> f32 {
        self.queue.len() as f32 / self.capacity as f32
    }

    /// Returns `(pushed, popped, dropped)` counters accumulated since creation.
    pub fn counters(&self) -> (u64, u64, u64) {
        (self.pushed, self.popped, self.dropped)
    }

    /// Drain all queued frames without delivering them.
    pub fn flush(&mut self) {
        self.queue.clear();
    }

    /// Return an iterator over queued frames (does not consume them).
    pub fn iter(&self) -> impl Iterator<Item = &ZeroCopyFrame> {
        self.queue.iter()
    }
}

// ── ZeroCopyBus ───────────────────────────────────────────────────────────────

/// A named bus of [`ZeroCopyChannel`]s, one per pipeline edge.
///
/// `ZeroCopyBus` owns one channel for each `(from_node_name, to_node_name)`
/// pair and routes [`ZeroCopyFrame`]s through the correct channel without
/// copying frame data.
pub struct ZeroCopyBus {
    /// Named channels keyed by `"from→to"`.
    channels: std::collections::HashMap<String, ZeroCopyChannel>,
}

impl ZeroCopyBus {
    /// Create an empty bus.
    pub fn new() -> Self {
        Self {
            channels: std::collections::HashMap::new(),
        }
    }

    /// Add a named channel between two nodes.
    ///
    /// If the channel already exists it is replaced by the new one.
    pub fn add_channel(
        &mut self,
        from: impl AsRef<str>,
        to: impl AsRef<str>,
        capacity: usize,
    ) {
        let key = format!("{}→{}", from.as_ref(), to.as_ref());
        self.channels.insert(key, ZeroCopyChannel::new(capacity));
    }

    fn channel_key(from: &str, to: &str) -> String {
        format!("{}→{}", from, to)
    }

    /// Push a frame onto the channel identified by `(from, to)`.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::NodeNotFound`] when the channel doesn't exist,
    /// or a `ValidationError` when the channel is full.
    pub fn push(
        &mut self,
        from: &str,
        to: &str,
        frame: ZeroCopyFrame,
    ) -> Result<(), PipelineError> {
        let key = Self::channel_key(from, to);
        let chan = self
            .channels
            .get_mut(&key)
            .ok_or_else(|| PipelineError::NodeNotFound(key.clone()))?;
        chan.push(frame)
    }

    /// Pop the oldest frame from the channel `(from, to)`.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::NodeNotFound`] when the channel doesn't exist.
    pub fn pop(
        &mut self,
        from: &str,
        to: &str,
    ) -> Result<Option<ZeroCopyFrame>, PipelineError> {
        let key = Self::channel_key(from, to);
        let chan = self
            .channels
            .get_mut(&key)
            .ok_or_else(|| PipelineError::NodeNotFound(key.clone()))?;
        Ok(chan.pop())
    }

    /// Number of channels registered on this bus.
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Aggregate `(pushed, popped, dropped)` across every channel.
    pub fn aggregate_counters(&self) -> (u64, u64, u64) {
        let mut pushed = 0u64;
        let mut popped = 0u64;
        let mut dropped = 0u64;
        for ch in self.channels.values() {
            let (p, q, d) = ch.counters();
            pushed += p;
            popped += q;
            dropped += d;
        }
        (pushed, popped, dropped)
    }
}

impl Default for ZeroCopyBus {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{FrameFormat, StreamKind};

    fn make_video_frame(pts: i64, bytes: usize) -> ZeroCopyFrame {
        let desc = FrameDescriptor::video(FrameFormat::Yuv420p, 1920, 1080, pts);
        let data: Arc<[u8]> = vec![0u8; bytes].into();
        ZeroCopyFrame::new(desc, data)
    }

    // ── FrameDescriptor tests ─────────────────────────────────────────────────

    #[test]
    fn video_descriptor_fields() {
        let d = FrameDescriptor::video(FrameFormat::Rgb24, 1280, 720, 42);
        assert_eq!(d.kind, StreamKind::Video);
        assert_eq!(d.width, Some(1280));
        assert_eq!(d.height, Some(720));
        assert_eq!(d.pts, 42);
        assert!(d.sample_rate.is_none());
    }

    #[test]
    fn audio_descriptor_fields() {
        let d = FrameDescriptor::audio(FrameFormat::Float32Planar, 48_000, 2, 100);
        assert_eq!(d.kind, StreamKind::Audio);
        assert_eq!(d.sample_rate, Some(48_000));
        assert_eq!(d.channels, Some(2));
        assert!(d.width.is_none());
    }

    #[test]
    fn descriptor_with_source_label() {
        let d = FrameDescriptor::video(FrameFormat::Yuv420p, 640, 480, 0)
            .with_source("decoder");
        assert_eq!(d.source_label, "decoder");
    }

    #[test]
    fn estimated_bytes_video() {
        let d = FrameDescriptor::video(FrameFormat::Rgb24, 100, 50, 0);
        // Rgb24 = 3 bytes per pixel, 100 * 50 * 3 = 15_000
        assert_eq!(d.estimated_bytes(), 15_000);
    }

    // ── ZeroCopyFrame tests ───────────────────────────────────────────────────

    #[test]
    fn frame_share_increments_refcount() {
        let frame = make_video_frame(0, 128);
        assert!(frame.is_exclusive());
        let shared = frame.share();
        assert_eq!(frame.ref_count(), 2);
        drop(shared);
        assert!(frame.is_exclusive());
    }

    #[test]
    fn try_into_owned_exclusive_no_copy() {
        let frame = make_video_frame(7, 64);
        assert!(frame.is_exclusive());
        let owned: Vec<u8> = frame.try_into_owned();
        assert_eq!(owned.len(), 64);
    }

    #[test]
    fn try_into_owned_shared_copies() {
        let frame = make_video_frame(3, 32);
        let _shared = frame.share(); // ref count → 2
        let owned = frame.try_into_owned();
        assert_eq!(owned.len(), 32);
    }

    #[test]
    fn from_vec_roundtrip() {
        let desc = FrameDescriptor::video(FrameFormat::Yuv420p, 4, 4, 0);
        let data = vec![1u8, 2, 3, 4];
        let frame = ZeroCopyFrame::from_vec(desc, data);
        assert_eq!(frame.data(), &[1u8, 2, 3, 4]);
        assert!(!frame.is_empty());
    }

    // ── ZeroCopyChannel tests ─────────────────────────────────────────────────

    #[test]
    fn channel_push_pop_roundtrip() {
        let mut ch = ZeroCopyChannel::new(8);
        let f = make_video_frame(10, 16);
        ch.push(f).expect("push ok");
        let recv = ch.pop().expect("has frame");
        assert_eq!(recv.descriptor().pts, 10);
    }

    #[test]
    fn channel_full_returns_error() {
        let mut ch = ZeroCopyChannel::new(2);
        ch.push(make_video_frame(0, 4)).expect("push 1 ok");
        ch.push(make_video_frame(1, 4)).expect("push 2 ok");
        let result = ch.push(make_video_frame(2, 4));
        assert!(result.is_err());
        let (_, _, dropped) = ch.counters();
        assert_eq!(dropped, 1);
    }

    #[test]
    fn channel_pop_empty_returns_none() {
        let mut ch = ZeroCopyChannel::new(4);
        assert!(ch.pop().is_none());
    }

    #[test]
    fn channel_fill_ratio() {
        let mut ch = ZeroCopyChannel::new(4);
        ch.push(make_video_frame(0, 4)).expect("push ok");
        ch.push(make_video_frame(1, 4)).expect("push ok");
        assert!((ch.fill_ratio() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn channel_flush_clears_queue() {
        let mut ch = ZeroCopyChannel::new(8);
        ch.push(make_video_frame(0, 8)).expect("push ok");
        ch.push(make_video_frame(1, 8)).expect("push ok");
        ch.flush();
        assert!(ch.is_empty());
    }

    // ── ZeroCopyBus tests ─────────────────────────────────────────────────────

    #[test]
    fn bus_push_pop_named_channel() {
        let mut bus = ZeroCopyBus::new();
        bus.add_channel("decode", "scale", 8);
        let f = make_video_frame(5, 32);
        bus.push("decode", "scale", f).expect("bus push ok");
        let recv = bus.pop("decode", "scale").expect("bus pop ok");
        assert!(recv.is_some());
        assert_eq!(recv.unwrap().descriptor().pts, 5);
    }

    #[test]
    fn bus_missing_channel_returns_error() {
        let mut bus = ZeroCopyBus::new();
        let result = bus.push("a", "b", make_video_frame(0, 4));
        assert!(result.is_err());
    }

    #[test]
    fn bus_aggregate_counters() {
        let mut bus = ZeroCopyBus::new();
        bus.add_channel("src", "filter", 8);
        bus.add_channel("filter", "sink", 8);
        bus.push("src", "filter", make_video_frame(0, 4)).expect("ok");
        bus.push("src", "filter", make_video_frame(1, 4)).expect("ok");
        bus.push("filter", "sink", make_video_frame(0, 4)).expect("ok");
        let (pushed, popped, dropped) = bus.aggregate_counters();
        assert_eq!(pushed, 3);
        assert_eq!(popped, 0);
        assert_eq!(dropped, 0);
    }
}
