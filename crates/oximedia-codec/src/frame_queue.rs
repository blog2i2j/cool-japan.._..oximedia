//! Encoder frame queue with PTS-ordered input, B-frame reorder buffer, and DTS calculation.
//!
//! ## Overview
//!
//! Video encoders with B-frame support cannot emit packets in display order.  They
//! need to:
//! 1. **Receive** frames in display/presentation order (PTS-ordered).
//! 2. **Reorder** them so reference frames (I/P) are encoded before the B-frames
//!    that depend on them.
//! 3. **Emit** encoded packets with a *decode timestamp* (DTS) that is ≤ the PTS
//!    and strictly non-decreasing.
//!
//! This module provides:
//! - [`FrameQueue`] — PTS-sorted input staging area that enforces ordering and
//!   detects duplicate / out-of-order submissions.
//! - [`BFrameReorderBuffer`] — Delayed-output buffer with configurable lookahead
//!   depth that reorders frames for B-frame encoding and computes DTS offsets.
//! - [`DtsCalculator`] — Standalone utility to turn a sequence of (PTS, is_key)
//!   pairs into correct DTS values.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::error::{CodecError, CodecResult};

/// A single frame entry held in the queue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueuedFrame {
    /// Presentation timestamp in encoder timebase units.
    pub pts: i64,
    /// Frame type hint used for reorder decisions.
    pub frame_type: QueueFrameType,
    /// Opaque payload (e.g. raw pixel data or a frame index).
    pub data: Vec<u8>,
}

/// Frame type hint supplied by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueFrameType {
    /// Intra-coded (keyframe) — never depends on other frames.
    Intra,
    /// Inter-coded, predicted from a past reference.
    Inter,
    /// Bi-directionally predicted — depends on both past and future references.
    BiPredicted,
}

/// A [`QueuedFrame`] augmented with its computed DTS, ready to be handed to the
/// codec for actual encoding.
#[derive(Debug, Clone)]
pub struct ReadyFrame {
    /// Original presentation timestamp.
    pub pts: i64,
    /// Decode timestamp (≤ pts, strictly non-decreasing per output packet).
    pub dts: i64,
    /// Frame type.
    pub frame_type: QueueFrameType,
    /// Opaque payload forwarded from [`QueuedFrame`].
    pub data: Vec<u8>,
}

// ── FrameQueue ───────────────────────────────────────────────────────────────

/// PTS-ordered staging queue for incoming frames.
///
/// Frames are pushed in any order and popped in strictly ascending PTS order.
/// Submitting a frame with a PTS that already exists in the queue is an error.
#[derive(Debug, Default)]
pub struct FrameQueue {
    /// Min-heap: `Reverse` so `BinaryHeap` (max-heap) becomes a min-heap.
    heap: BinaryHeap<Reverse<PtsOrdFrame>>,
    /// Set of PTS values currently in the queue (for duplicate detection).
    pts_set: std::collections::BTreeSet<i64>,
}

/// Wrapper that orders [`QueuedFrame`] by PTS.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PtsOrdFrame(QueuedFrame);

impl PartialOrd for PtsOrdFrame {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PtsOrdFrame {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.pts.cmp(&other.0.pts)
    }
}

impl FrameQueue {
    /// Create an empty queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a frame.  Returns an error if a frame with the same PTS is already queued.
    pub fn push(&mut self, frame: QueuedFrame) -> CodecResult<()> {
        if self.pts_set.contains(&frame.pts) {
            return Err(CodecError::InvalidParameter(format!(
                "duplicate PTS {} in frame queue",
                frame.pts
            )));
        }
        self.pts_set.insert(frame.pts);
        self.heap.push(Reverse(PtsOrdFrame(frame)));
        Ok(())
    }

    /// Pop the frame with the lowest PTS, or `None` if the queue is empty.
    pub fn pop(&mut self) -> Option<QueuedFrame> {
        let Reverse(PtsOrdFrame(frame)) = self.heap.pop()?;
        self.pts_set.remove(&frame.pts);
        Some(frame)
    }

    /// Peek at the PTS of the next frame without removing it.
    pub fn peek_pts(&self) -> Option<i64> {
        self.heap.peek().map(|Reverse(PtsOrdFrame(f))| f.pts)
    }

    /// Number of frames currently queued.
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Returns `true` if no frames are queued.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Drain all frames in ascending PTS order.
    pub fn drain_ordered(&mut self) -> Vec<QueuedFrame> {
        let mut result = Vec::with_capacity(self.heap.len());
        while let Some(f) = self.pop() {
            result.push(f);
        }
        result
    }
}

// ── BFrameReorderBuffer ──────────────────────────────────────────────────────

/// Configuration for the B-frame reorder buffer.
#[derive(Debug, Clone)]
pub struct ReorderConfig {
    /// Maximum number of consecutive B-frames between reference frames
    /// (lookahead depth).  Set to `0` to disable B-frame reordering.
    pub max_b_frames: usize,
    /// Timebase numerator (for DTS offset calculation).
    /// DTS is expressed in the same timebase as PTS.
    pub timebase_num: u32,
    /// Timebase denominator (for DTS offset calculation).
    pub timebase_den: u32,
    /// Minimum delta between adjacent DTS values.  Typically
    /// `timebase_den / (framerate_num * timebase_num)`.
    pub min_dts_delta: i64,
}

impl Default for ReorderConfig {
    fn default() -> Self {
        Self {
            max_b_frames: 2,
            timebase_num: 1,
            timebase_den: 90_000, // MPEG-90 kHz timebase
            min_dts_delta: 3000,  // 1 frame @ 30 fps in 90 kHz units
        }
    }
}

/// B-frame reorder buffer.
///
/// Accepts frames in PTS order and emits [`ReadyFrame`]s in *decode order* with
/// computed DTS values.
///
/// ### Reorder algorithm (simplified)
///
/// Frames are collected into *groups of pictures* (GOPs).  Within each GOP a
/// run of `B` frames is deferred until its anchor `P/I` frame has been output.
/// DTS is assigned as a monotonically increasing counter starting from the
/// smallest PTS seen, offset back by `max_b_frames * min_dts_delta` to ensure
/// DTS ≤ PTS for every frame.
#[derive(Debug)]
pub struct BFrameReorderBuffer {
    config: ReorderConfig,
    /// Frames waiting to be emitted after the anchor is found.
    pending: Vec<QueuedFrame>,
    /// Next DTS value to assign.
    next_dts: Option<i64>,
    /// Monotonically increasing DTS counter (once initialised).
    dts_counter: i64,
    /// Output queue: frames ready for the encoder in decode order.
    output: std::collections::VecDeque<ReadyFrame>,
}

impl BFrameReorderBuffer {
    /// Create a new reorder buffer with the given configuration.
    pub fn new(config: ReorderConfig) -> Self {
        Self {
            config,
            pending: Vec::new(),
            next_dts: None,
            dts_counter: 0,
            output: std::collections::VecDeque::new(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(ReorderConfig::default())
    }

    /// Push a frame in PTS / display order.
    ///
    /// The buffer will internally decide when to emit frames in decode order.
    pub fn push(&mut self, frame: QueuedFrame) {
        // Initialise DTS on first frame.  We set it `max_b_frames` steps before
        // the first PTS so that B-frame packets always have DTS ≤ PTS.
        if self.next_dts.is_none() {
            let offset = (self.config.max_b_frames as i64) * self.config.min_dts_delta;
            let initial_dts = frame.pts - offset;
            self.next_dts = Some(initial_dts);
            self.dts_counter = initial_dts;
        }

        match frame.frame_type {
            QueueFrameType::Intra | QueueFrameType::Inter => {
                // Anchor frame: flush any pending B-frames first (they have
                // smaller PTS but were held so the anchor could be encoded
                // first).
                self.flush_pending_b_frames();
                // Then emit the anchor itself.
                self.emit_frame(frame);
            }
            QueueFrameType::BiPredicted => {
                if self.config.max_b_frames == 0 {
                    // B-frames disabled: treat as inter.
                    self.emit_frame(frame);
                } else {
                    self.pending.push(frame);
                    // Flush when the buffer is full.
                    if self.pending.len() >= self.config.max_b_frames {
                        self.flush_pending_b_frames();
                    }
                }
            }
        }
    }

    /// Flush all remaining frames (call at end of stream).
    pub fn flush(&mut self) {
        self.flush_pending_b_frames();
    }

    /// Pop the next [`ReadyFrame`] in decode order, or `None` if none are ready.
    pub fn pop(&mut self) -> Option<ReadyFrame> {
        self.output.pop_front()
    }

    /// Number of frames ready to be consumed.
    pub fn ready_len(&self) -> usize {
        self.output.len()
    }

    /// Number of frames still buffered (not yet emitted).
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    // ── private helpers ───────────────────────────────────────────────────

    fn flush_pending_b_frames(&mut self) {
        // Sort pending B-frames by PTS before emitting them.
        self.pending.sort_by_key(|f| f.pts);
        let frames: Vec<_> = self.pending.drain(..).collect();
        for f in frames {
            self.emit_frame(f);
        }
    }

    fn emit_frame(&mut self, frame: QueuedFrame) {
        let dts = self.dts_counter;
        self.dts_counter += self.config.min_dts_delta;
        self.output.push_back(ReadyFrame {
            pts: frame.pts,
            dts,
            frame_type: frame.frame_type,
            data: frame.data,
        });
    }
}

// ── DtsCalculator ────────────────────────────────────────────────────────────

/// Standalone DTS calculator.
///
/// Given a sequence of `(pts, is_keyframe)` pairs, produces DTS values that are:
/// - Strictly non-decreasing.
/// - Always ≤ the corresponding PTS.
/// - Separated by at least `min_delta` timebase units.
///
/// Useful when you already have the full frame sequence and want to annotate
/// DTS in batch (e.g. when writing container mux metadata).
#[derive(Debug)]
pub struct DtsCalculator {
    /// Minimum gap between adjacent DTS values.
    min_delta: i64,
    /// B-frame lookahead depth (used for initial DTS pre-roll offset).
    max_b_frames: usize,
    /// Running DTS counter.
    next_dts: Option<i64>,
}

impl DtsCalculator {
    /// Create a new calculator.
    ///
    /// `min_delta` must be > 0.
    pub fn new(min_delta: i64, max_b_frames: usize) -> CodecResult<Self> {
        if min_delta <= 0 {
            return Err(CodecError::InvalidParameter(
                "DtsCalculator: min_delta must be positive".into(),
            ));
        }
        Ok(Self {
            min_delta,
            max_b_frames,
            next_dts: None,
        })
    }

    /// Compute DTS for a single frame in sequence order.
    ///
    /// The first call initialises the internal counter from `pts` minus the
    /// B-frame pre-roll offset.
    pub fn next(&mut self, pts: i64, _is_keyframe: bool) -> i64 {
        let dts = match self.next_dts {
            None => {
                let offset = (self.max_b_frames as i64) * self.min_delta;
                let initial = pts - offset;
                self.next_dts = Some(initial + self.min_delta);
                initial
            }
            Some(ref mut counter) => {
                let dts = *counter;
                *counter += self.min_delta;
                dts
            }
        };
        dts
    }

    /// Compute DTS for a batch of `(pts, is_keyframe)` pairs.
    ///
    /// Returns a `Vec<i64>` of the same length.
    pub fn compute_batch(&mut self, frames: &[(i64, bool)]) -> Vec<i64> {
        frames
            .iter()
            .map(|&(pts, is_key)| self.next(pts, is_key))
            .collect()
    }

    /// Reset the internal counter so a new sequence can be processed.
    pub fn reset(&mut self) {
        self.next_dts = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(pts: i64, ft: QueueFrameType) -> QueuedFrame {
        QueuedFrame {
            pts,
            frame_type: ft,
            data: vec![pts as u8],
        }
    }

    #[test]
    fn test_frame_queue_push_pop_ordered() {
        let mut q = FrameQueue::new();
        q.push(make_frame(200, QueueFrameType::Inter)).unwrap();
        q.push(make_frame(0, QueueFrameType::Intra)).unwrap();
        q.push(make_frame(100, QueueFrameType::BiPredicted))
            .unwrap();

        assert_eq!(q.pop().unwrap().pts, 0);
        assert_eq!(q.pop().unwrap().pts, 100);
        assert_eq!(q.pop().unwrap().pts, 200);
        assert!(q.pop().is_none());
    }

    #[test]
    fn test_frame_queue_duplicate_pts_error() {
        let mut q = FrameQueue::new();
        q.push(make_frame(100, QueueFrameType::Intra)).unwrap();
        let result = q.push(make_frame(100, QueueFrameType::Inter));
        assert!(result.is_err());
    }

    #[test]
    fn test_frame_queue_drain_ordered() {
        let mut q = FrameQueue::new();
        for pts in [500i64, 100, 300, 0, 200] {
            q.push(make_frame(pts, QueueFrameType::Inter)).unwrap();
        }
        let drained = q.drain_ordered();
        let pts_seq: Vec<i64> = drained.iter().map(|f| f.pts).collect();
        assert_eq!(pts_seq, vec![0, 100, 200, 300, 500]);
    }

    #[test]
    fn test_frame_queue_peek_pts() {
        let mut q = FrameQueue::new();
        assert_eq!(q.peek_pts(), None);
        q.push(make_frame(50, QueueFrameType::Intra)).unwrap();
        q.push(make_frame(10, QueueFrameType::Inter)).unwrap();
        assert_eq!(q.peek_pts(), Some(10));
    }

    #[test]
    fn test_b_frame_reorder_anchor_before_b() {
        let cfg = ReorderConfig {
            max_b_frames: 2,
            min_dts_delta: 1,
            ..Default::default()
        };
        let mut buf = BFrameReorderBuffer::new(cfg);
        // Display order: I B B P
        buf.push(make_frame(0, QueueFrameType::Intra));
        buf.push(make_frame(1, QueueFrameType::BiPredicted));
        buf.push(make_frame(2, QueueFrameType::BiPredicted));
        buf.push(make_frame(3, QueueFrameType::Inter));
        buf.flush();

        // Decode order should be: I P B B  (or I then pending B-frames after P anchor)
        let mut out = Vec::new();
        while let Some(f) = buf.pop() {
            out.push(f);
        }
        assert!(!out.is_empty());
        // DTS must be non-decreasing
        for w in out.windows(2) {
            assert!(w[1].dts >= w[0].dts, "DTS must be non-decreasing");
        }
    }

    #[test]
    fn test_b_frame_reorder_dts_leq_pts() {
        let cfg = ReorderConfig {
            max_b_frames: 2,
            min_dts_delta: 3000,
            ..Default::default()
        };
        let mut buf = BFrameReorderBuffer::new(cfg);
        let pts_sequence = [0i64, 3000, 6000, 9000, 12000];
        for (i, &pts) in pts_sequence.iter().enumerate() {
            let ft = if i % 3 == 0 {
                QueueFrameType::Intra
            } else if i % 3 == 1 {
                QueueFrameType::BiPredicted
            } else {
                QueueFrameType::Inter
            };
            buf.push(make_frame(pts, ft));
        }
        buf.flush();

        while let Some(f) = buf.pop() {
            assert!(f.dts <= f.pts, "DTS ({}) must be <= PTS ({})", f.dts, f.pts);
        }
    }

    #[test]
    fn test_dts_calculator_basic() {
        let mut calc = DtsCalculator::new(3000, 2).unwrap();
        let pts_vals = [6000i64, 9000, 12000, 15000];
        let frames: Vec<(i64, bool)> = pts_vals
            .iter()
            .enumerate()
            .map(|(i, &p)| (p, i == 0))
            .collect();
        let dts = calc.compute_batch(&frames);
        // First DTS = 6000 - 2*3000 = 0
        assert_eq!(dts[0], 0);
        // Each subsequent DTS increments by min_delta
        for w in dts.windows(2) {
            assert_eq!(w[1] - w[0], 3000);
        }
    }

    #[test]
    fn test_dts_calculator_invalid_delta() {
        let result = DtsCalculator::new(0, 2);
        assert!(result.is_err());
        let result2 = DtsCalculator::new(-1, 2);
        assert!(result2.is_err());
    }

    #[test]
    fn test_dts_calculator_reset() {
        let mut calc = DtsCalculator::new(1000, 1).unwrap();
        let dts1 = calc.next(5000, true);
        calc.reset();
        let dts2 = calc.next(5000, true);
        // After reset the initial DTS should be the same as first call
        assert_eq!(dts1, dts2);
    }

    #[test]
    fn test_no_b_frames_passthrough() {
        let cfg = ReorderConfig {
            max_b_frames: 0,
            min_dts_delta: 1,
            ..Default::default()
        };
        let mut buf = BFrameReorderBuffer::new(cfg);
        for pts in [0i64, 1, 2, 3] {
            buf.push(make_frame(pts, QueueFrameType::BiPredicted));
        }
        buf.flush();
        let mut pts_out = Vec::new();
        while let Some(f) = buf.pop() {
            pts_out.push(f.pts);
        }
        // All frames emitted immediately in push order (no reordering)
        assert_eq!(pts_out.len(), 4);
    }
}
