//! Backpressure and flow control between pipeline stages.
//!
//! This module provides bounded inter-stage queues and signalling primitives
//! that let a slow downstream stage exert back-pressure on fast upstream
//! producers, preventing unbounded memory growth during pipeline execution.
//!
//! # Design
//!
//! Each stage-to-stage connection is modelled as a [`BoundedStageQueue`].
//! A [`BackpressureController`] owns one queue per pipeline edge and exposes
//! a unified API for producers and consumers:
//!
//! - **Producer**: calls [`BackpressureController::push`]; if the queue is
//!   full the call returns [`BackpressureSignal::ShouldThrottle`] so the
//!   caller knows to slow down or drop frames.
//! - **Consumer**: calls [`BackpressureController::pop`]; returns `None` when
//!   the queue is empty.
//!
//! Dropped frame counts are tracked per edge so [`PipelineMetrics`] can later
//! report them accurately.
//!
//! # Example
//!
//! ```rust
//! use oximedia_pipeline::backpressure::{BackpressureController, BackpressureSignal, QueueId};
//!
//! let mut ctrl = BackpressureController::new();
//! let qid = ctrl.add_queue(QueueId::new("scale", "sink"), 4);
//!
//! // Producer: push a payload
//! let signal = ctrl.push(qid, 42u64).expect("push ok");
//! assert_eq!(signal, BackpressureSignal::Ok);
//!
//! // Consumer: pop it back out
//! let value = ctrl.pop::<u64>(qid).expect("pop ok");
//! assert_eq!(value, Some(42u64));
//! ```

use std::any::Any;
use std::collections::{HashMap, VecDeque};

use crate::PipelineError;

// ── QueueId ───────────────────────────────────────────────────────────────────

/// Identifies a single inter-stage queue by its upstream node name and
/// downstream node name.  Two queues on different edges are always distinct
/// even when they share the same upstream node.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QueueId {
    /// Human-readable upstream stage name.
    pub upstream: String,
    /// Human-readable downstream stage name.
    pub downstream: String,
}

impl QueueId {
    /// Create a new `QueueId` from any string-like upstream / downstream pair.
    pub fn new(upstream: impl Into<String>, downstream: impl Into<String>) -> Self {
        Self {
            upstream: upstream.into(),
            downstream: downstream.into(),
        }
    }
}

impl std::fmt::Display for QueueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}→{}", self.upstream, self.downstream)
    }
}

// ── BackpressureSignal ────────────────────────────────────────────────────────

/// The outcome of a [`BackpressureController::push`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackpressureSignal {
    /// The frame was enqueued successfully; the producer may continue at
    /// its normal rate.
    Ok,
    /// The queue is at or above the high-water mark.  The producer should
    /// reduce its output rate or skip non-essential frames.
    ShouldThrottle,
    /// The queue is completely full.  The frame was **not** enqueued; the
    /// producer must drop this frame or block until space is available.
    QueueFull,
}

// ── QueueStats ────────────────────────────────────────────────────────────────

/// Aggregate statistics for a single inter-stage queue.
#[derive(Debug, Clone, Default)]
pub struct QueueStats {
    /// Total number of frames successfully enqueued over the queue's lifetime.
    pub total_pushed: u64,
    /// Total number of frames successfully dequeued over the queue's lifetime.
    pub total_popped: u64,
    /// Number of times a push was rejected because the queue was full.
    pub dropped_frames: u64,
    /// Maximum observed queue depth (high-water mark).
    pub peak_depth: usize,
    /// Number of times `ShouldThrottle` was returned to the producer.
    pub throttle_events: u64,
}

// ── BoundedStageQueue ─────────────────────────────────────────────────────────

/// A type-erased bounded FIFO queue for a single pipeline edge.
struct BoundedStageQueue {
    /// Maximum number of frames the queue may hold at once.
    capacity: usize,
    /// High-water mark as a fraction of capacity (0.0–1.0).  When `depth / capacity`
    /// exceeds this, `ShouldThrottle` is returned even though the push succeeded.
    high_water_ratio: f32,
    /// The queue itself (type-erased payloads).
    items: VecDeque<Box<dyn Any + Send + 'static>>,
    /// Running statistics for this queue.
    stats: QueueStats,
}

impl BoundedStageQueue {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            high_water_ratio: 0.75,
            items: VecDeque::with_capacity(capacity),
            stats: QueueStats::default(),
        }
    }

    /// Push a type-erased payload.  Returns the appropriate [`BackpressureSignal`].
    fn push_erased(&mut self, item: Box<dyn Any + Send + 'static>) -> BackpressureSignal {
        if self.items.len() >= self.capacity {
            self.stats.dropped_frames += 1;
            return BackpressureSignal::QueueFull;
        }
        self.items.push_back(item);
        self.stats.total_pushed += 1;
        let depth = self.items.len();
        if depth > self.stats.peak_depth {
            self.stats.peak_depth = depth;
        }
        let ratio = depth as f32 / self.capacity as f32;
        if ratio >= self.high_water_ratio {
            self.stats.throttle_events += 1;
            BackpressureSignal::ShouldThrottle
        } else {
            BackpressureSignal::Ok
        }
    }

    /// Pop a type-erased payload.
    fn pop_erased(&mut self) -> Option<Box<dyn Any + Send + 'static>> {
        let item = self.items.pop_front();
        if item.is_some() {
            self.stats.total_popped += 1;
        }
        item
    }

    /// Current number of items in the queue.
    fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns `true` when the queue is empty.
    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

// ── Handle ────────────────────────────────────────────────────────────────────

/// An opaque handle returned by [`BackpressureController::add_queue`] and used
/// to address a specific queue in subsequent push/pop calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueueHandle(usize);

// ── BackpressureController ────────────────────────────────────────────────────

/// Manages a collection of bounded inter-stage queues for a pipeline.
///
/// `BackpressureController` is the central coordination point: it owns every
/// queue, assigns handles to callers, and forwards push/pop operations to the
/// correct queue while accumulating statistics.
pub struct BackpressureController {
    /// The queues, stored in insertion order.  The handle's inner index is
    /// the position in this `Vec`.
    queues: Vec<BoundedStageQueue>,
    /// Maps human-readable [`QueueId`] to the `Vec` index / [`QueueHandle`].
    id_to_handle: HashMap<QueueId, QueueHandle>,
    /// Reverse map for diagnostics.
    handle_to_id: HashMap<QueueHandle, QueueId>,
}

impl BackpressureController {
    /// Create an empty controller with no queues.
    pub fn new() -> Self {
        Self {
            queues: Vec::new(),
            id_to_handle: HashMap::new(),
            handle_to_id: HashMap::new(),
        }
    }

    /// Register a new bounded queue between two stages.
    ///
    /// Returns a [`QueueHandle`] that the caller must retain for push/pop
    /// operations.  If a queue with the same `QueueId` already exists, the
    /// existing handle is returned without creating a duplicate.
    pub fn add_queue(&mut self, id: QueueId, capacity: usize) -> QueueHandle {
        if let Some(&existing) = self.id_to_handle.get(&id) {
            return existing;
        }
        let handle = QueueHandle(self.queues.len());
        self.queues.push(BoundedStageQueue::new(capacity.max(1)));
        self.id_to_handle.insert(id.clone(), handle);
        self.handle_to_id.insert(handle, id);
        handle
    }

    /// Set the high-water ratio (0.0–1.0) for the queue identified by `handle`.
    ///
    /// When the fill level reaches `ratio × capacity`, future pushes return
    /// [`BackpressureSignal::ShouldThrottle`] until the consumer drains the
    /// queue below the threshold.
    ///
    /// The default ratio is `0.75`.
    pub fn set_high_water_ratio(
        &mut self,
        handle: QueueHandle,
        ratio: f32,
    ) -> Result<(), PipelineError> {
        let q = self
            .queues
            .get_mut(handle.0)
            .ok_or_else(|| PipelineError::ValidationError(format!("invalid queue handle {}", handle.0)))?;
        let clamped = ratio.clamp(0.0, 1.0);
        q.high_water_ratio = clamped;
        Ok(())
    }

    /// Push a typed payload onto the queue identified by `handle`.
    ///
    /// The payload is type-erased internally; callers must use the same type `T`
    /// for both push and pop on a given queue.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::ValidationError`] when `handle` is invalid.
    pub fn push<T: Send + 'static>(
        &mut self,
        handle: QueueHandle,
        item: T,
    ) -> Result<BackpressureSignal, PipelineError> {
        let q = self
            .queues
            .get_mut(handle.0)
            .ok_or_else(|| PipelineError::ValidationError(format!("invalid queue handle {}", handle.0)))?;
        Ok(q.push_erased(Box::new(item)))
    }

    /// Pop the oldest payload from the queue identified by `handle`, returning
    /// `None` when the queue is empty.
    ///
    /// # Errors
    ///
    /// - Returns [`PipelineError::ValidationError`] when `handle` is invalid.
    /// - Returns [`PipelineError::ValidationError`] when the stored item does
    ///   not downcast to `T` (type mismatch between producer and consumer).
    pub fn pop<T: Send + 'static>(
        &mut self,
        handle: QueueHandle,
    ) -> Result<Option<T>, PipelineError> {
        let q = self
            .queues
            .get_mut(handle.0)
            .ok_or_else(|| PipelineError::ValidationError(format!("invalid queue handle {}", handle.0)))?;
        match q.pop_erased() {
            None => Ok(None),
            Some(boxed) => {
                boxed
                    .downcast::<T>()
                    .map(|b| Some(*b))
                    .map_err(|_| PipelineError::ValidationError(
                        "type mismatch: pushed and popped types differ".to_string(),
                    ))
            }
        }
    }

    /// Return the current fill depth of the queue.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::ValidationError`] when `handle` is invalid.
    pub fn depth(&self, handle: QueueHandle) -> Result<usize, PipelineError> {
        self.queues
            .get(handle.0)
            .map(|q| q.len())
            .ok_or_else(|| PipelineError::ValidationError(format!("invalid queue handle {}", handle.0)))
    }

    /// Returns `true` when the queue is empty.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::ValidationError`] when `handle` is invalid.
    pub fn is_empty(&self, handle: QueueHandle) -> Result<bool, PipelineError> {
        self.queues
            .get(handle.0)
            .map(|q| q.is_empty())
            .ok_or_else(|| PipelineError::ValidationError(format!("invalid queue handle {}", handle.0)))
    }

    /// Return a snapshot of statistics for the given queue.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::ValidationError`] when `handle` is invalid.
    pub fn stats(&self, handle: QueueHandle) -> Result<QueueStats, PipelineError> {
        self.queues
            .get(handle.0)
            .map(|q| q.stats.clone())
            .ok_or_else(|| PipelineError::ValidationError(format!("invalid queue handle {}", handle.0)))
    }

    /// Return the [`QueueId`] for a given handle, if it exists.
    pub fn id_for_handle(&self, handle: QueueHandle) -> Option<&QueueId> {
        self.handle_to_id.get(&handle)
    }

    /// Return the handle for a given [`QueueId`], if the queue was registered.
    pub fn handle_for_id(&self, id: &QueueId) -> Option<QueueHandle> {
        self.id_to_handle.get(id).copied()
    }

    /// Return the capacity of the queue identified by `handle`.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::ValidationError`] when `handle` is invalid.
    pub fn capacity(&self, handle: QueueHandle) -> Result<usize, PipelineError> {
        self.queues
            .get(handle.0)
            .map(|q| q.capacity)
            .ok_or_else(|| PipelineError::ValidationError(format!("invalid queue handle {}", handle.0)))
    }

    /// Total number of queues registered with this controller.
    pub fn queue_count(&self) -> usize {
        self.queues.len()
    }

    /// Drain **all** items from every queue (e.g. on pipeline reset).
    pub fn flush_all(&mut self) {
        for q in &mut self.queues {
            q.items.clear();
        }
    }

    /// Drain all items from a single queue.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::ValidationError`] when `handle` is invalid.
    pub fn flush(&mut self, handle: QueueHandle) -> Result<(), PipelineError> {
        let q = self
            .queues
            .get_mut(handle.0)
            .ok_or_else(|| PipelineError::ValidationError(format!("invalid queue handle {}", handle.0)))?;
        q.items.clear();
        Ok(())
    }

    /// Aggregate statistics across every registered queue.
    ///
    /// Returns `(total_pushed, total_popped, total_dropped)`.
    pub fn aggregate_stats(&self) -> (u64, u64, u64) {
        let mut pushed = 0u64;
        let mut popped = 0u64;
        let mut dropped = 0u64;
        for q in &self.queues {
            pushed += q.stats.total_pushed;
            popped += q.stats.total_popped;
            dropped += q.stats.dropped_frames;
        }
        (pushed, popped, dropped)
    }
}

impl Default for BackpressureController {
    fn default() -> Self {
        Self::new()
    }
}

// ── FlowPolicy ────────────────────────────────────────────────────────────────

/// Determines how a producer should respond when a downstream queue signals
/// throttling or fullness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowPolicy {
    /// Drop the current frame and move on to the next one immediately.
    DropFrame,
    /// Repeat the previous frame to fill the slot without introducing new data.
    RepeatFrame,
    /// Block the current thread until the queue drains below capacity.
    Block,
    /// Pass frames through regardless of queue pressure (disables back-pressure).
    /// The queue will drop frames when full rather than signalling the producer.
    Passthrough,
}

// ── StageFlowConfig ───────────────────────────────────────────────────────────

/// Per-stage flow control configuration stored alongside the
/// [`BackpressureController`] to guide the pipeline executor.
#[derive(Debug, Clone)]
pub struct StageFlowConfig {
    /// The human-readable stage name this config applies to.
    pub stage_name: String,
    /// How the stage should respond to downstream back-pressure.
    pub policy: FlowPolicy,
    /// Maximum number of consecutive dropped frames before the pipeline raises
    /// an error (0 = unlimited).
    pub max_consecutive_drops: u32,
    /// When `Some(n)`, the executor introduces an artificial delay of `n`
    /// microseconds between stage activations to smooth out rate mismatches.
    pub pace_us: Option<u64>,
}

impl StageFlowConfig {
    /// Create a stage config with the [`FlowPolicy::DropFrame`] policy.
    pub fn drop_policy(stage_name: impl Into<String>) -> Self {
        Self {
            stage_name: stage_name.into(),
            policy: FlowPolicy::DropFrame,
            max_consecutive_drops: 0,
            pace_us: None,
        }
    }

    /// Create a stage config with the [`FlowPolicy::Block`] policy.
    pub fn block_policy(stage_name: impl Into<String>) -> Self {
        Self {
            stage_name: stage_name.into(),
            policy: FlowPolicy::Block,
            max_consecutive_drops: 0,
            pace_us: None,
        }
    }

    /// Create a stage config with the [`FlowPolicy::Passthrough`] policy.
    pub fn passthrough(stage_name: impl Into<String>) -> Self {
        Self {
            stage_name: stage_name.into(),
            policy: FlowPolicy::Passthrough,
            max_consecutive_drops: 0,
            pace_us: None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctrl_with_queue(cap: usize) -> (BackpressureController, QueueHandle) {
        let mut ctrl = BackpressureController::new();
        let qid = QueueId::new("source", "sink");
        let handle = ctrl.add_queue(qid, cap);
        (ctrl, handle)
    }

    #[test]
    fn push_pop_round_trip() {
        let (mut ctrl, h) = make_ctrl_with_queue(8);
        let sig = ctrl.push(h, 99u32).expect("push to valid queue should succeed");
        assert_eq!(sig, BackpressureSignal::Ok);
        let val = ctrl.pop::<u32>(h).expect("pop from valid queue should succeed");
        assert_eq!(val, Some(99u32));
    }

    #[test]
    fn pop_empty_queue_returns_none() {
        let (mut ctrl, h) = make_ctrl_with_queue(4);
        let val = ctrl.pop::<u8>(h).expect("pop from valid empty queue should succeed");
        assert!(val.is_none());
    }

    #[test]
    fn queue_full_signals_drop() {
        let (mut ctrl, h) = make_ctrl_with_queue(2);
        ctrl.push(h, 1u8).expect("push 1 should succeed");
        ctrl.push(h, 2u8).expect("push 2 should succeed");
        // Queue is now at capacity
        let sig = ctrl.push(h, 3u8).expect("push to full queue returns signal, not error");
        assert_eq!(sig, BackpressureSignal::QueueFull);
        let stats = ctrl.stats(h).expect("stats should succeed for valid handle");
        assert_eq!(stats.dropped_frames, 1);
        assert_eq!(stats.total_pushed, 2);
    }

    #[test]
    fn high_water_mark_signals_throttle() {
        let (mut ctrl, h) = make_ctrl_with_queue(4);
        // Default high-water ratio = 0.75, so throttle fires at depth >= 3
        ctrl.push(h, 0u32).expect("push 0 should succeed");
        ctrl.push(h, 1u32).expect("push 1 should succeed");
        let sig = ctrl.push(h, 2u32).expect("push 2 returns signal, not error");
        assert_eq!(sig, BackpressureSignal::ShouldThrottle);
        let stats = ctrl.stats(h).expect("stats should succeed for valid handle");
        assert_eq!(stats.throttle_events, 1);
    }

    #[test]
    fn aggregate_stats_accumulate_across_queues() {
        let mut ctrl = BackpressureController::new();
        let h1 = ctrl.add_queue(QueueId::new("a", "b"), 8);
        let h2 = ctrl.add_queue(QueueId::new("b", "c"), 8);
        ctrl.push(h1, 1u32).expect("push to h1 should succeed");
        ctrl.push(h1, 2u32).expect("push to h1 should succeed");
        ctrl.push(h2, 3u32).expect("push to h2 should succeed");
        ctrl.pop::<u32>(h1).expect("pop from h1 should succeed");
        let (pushed, popped, dropped) = ctrl.aggregate_stats();
        assert_eq!(pushed, 3);
        assert_eq!(popped, 1);
        assert_eq!(dropped, 0);
    }

    #[test]
    fn flush_clears_queue() {
        let (mut ctrl, h) = make_ctrl_with_queue(8);
        ctrl.push(h, 42u64).expect("push should succeed");
        ctrl.push(h, 43u64).expect("push should succeed");
        assert_eq!(ctrl.depth(h).expect("depth should succeed for valid handle"), 2);
        ctrl.flush(h).expect("flush should succeed for valid handle");
        assert!(ctrl.is_empty(h).expect("is_empty should succeed for valid handle"));
    }

    #[test]
    fn add_same_queue_id_returns_existing_handle() {
        let mut ctrl = BackpressureController::new();
        let qid = QueueId::new("x", "y");
        let h1 = ctrl.add_queue(qid.clone(), 4);
        let h2 = ctrl.add_queue(qid, 8);
        assert_eq!(h1, h2);
        assert_eq!(ctrl.queue_count(), 1);
    }

    #[test]
    fn set_high_water_ratio_clamped() {
        let (mut ctrl, h) = make_ctrl_with_queue(10);
        ctrl.set_high_water_ratio(h, 1.5).expect("set_high_water_ratio should succeed for valid handle"); // should clamp to 1.0
        // With ratio=1.0 only a completely full queue triggers throttle;
        // pushing 9 items (depth 1..9, ratio < 1.0) should all return Ok.
        for i in 0u32..9 {
            let sig = ctrl.push(h, i).expect("push should succeed for item {i}");
            assert_eq!(sig, BackpressureSignal::Ok, "item {i} unexpectedly throttled");
        }
        // The 10th push fills the queue exactly (depth/capacity == 1.0 >= 1.0)
        // so it returns ShouldThrottle (not QueueFull — it was still enqueued)
        let sig = ctrl.push(h, 9u32).expect("10th push should succeed and return signal");
        assert_eq!(sig, BackpressureSignal::ShouldThrottle);
    }

    #[test]
    fn type_mismatch_returns_error() {
        let (mut ctrl, h) = make_ctrl_with_queue(4);
        ctrl.push(h, 42u32).expect("push u32 should succeed");
        // Attempt to pop as a different type
        let result = ctrl.pop::<String>(h);
        assert!(result.is_err());
    }

    #[test]
    fn flow_policy_variants_constructors() {
        let drop = StageFlowConfig::drop_policy("encoder");
        assert_eq!(drop.policy, FlowPolicy::DropFrame);
        let block = StageFlowConfig::block_policy("decoder");
        assert_eq!(block.policy, FlowPolicy::Block);
        let pass = StageFlowConfig::passthrough("mux");
        assert_eq!(pass.policy, FlowPolicy::Passthrough);
    }

    #[test]
    fn peak_depth_tracked() {
        let (mut ctrl, h) = make_ctrl_with_queue(16);
        for i in 0u32..5 {
            ctrl.push(h, i).expect("push should succeed");
        }
        // Drain 3
        ctrl.pop::<u32>(h).expect("pop 1 should succeed");
        ctrl.pop::<u32>(h).expect("pop 2 should succeed");
        ctrl.pop::<u32>(h).expect("pop 3 should succeed");
        let stats = ctrl.stats(h).expect("stats should succeed for valid handle");
        assert_eq!(stats.peak_depth, 5);
        assert_eq!(stats.total_popped, 3);
    }
}
