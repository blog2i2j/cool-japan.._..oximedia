//! Operation batching for collaborative sync messages.
//!
//! Real-time collaboration sessions can generate hundreds of tiny operations
//! per second (cursor moves, incremental keystrokes, parameter tweaks).
//! Sending each operation as an individual network message is expensive in
//! both bandwidth and connection overhead.  This module provides an
//! [`OpBatcher`] that accumulates operations into *batches* that are flushed
//! either when the batch reaches a size threshold or after a configurable idle
//! timeout.
//!
//! # Key types
//! * [`BatchedOp`] — a single operation tagged with session/user metadata.
//! * [`OpBatch`] — an ordered, immutable snapshot of accumulated operations.
//! * [`OpBatcher`] — the mutable accumulator.
//! * [`BatchStats`] — aggregate metrics across all batches produced so far.

use crate::operation_log::{OpType, Operation};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

// ─────────────────────────────────────────────────────────────────────────────
// Domain types
// ─────────────────────────────────────────────────────────────────────────────

/// A single operation enriched with session context for over-the-wire
/// transmission.
#[derive(Debug, Clone)]
pub struct BatchedOp {
    /// The underlying operation.
    pub op: Operation,
    /// Session identifier (opaque string).
    pub session_id: String,
    /// User identifier.
    pub user_id: u32,
    /// Wall-clock time the op was queued (for latency tracking).
    pub queued_at: Instant,
}

impl BatchedOp {
    /// Create a new `BatchedOp` stamping `queued_at` to `now`.
    pub fn new(op: Operation, session_id: impl Into<String>, user_id: u32) -> Self {
        Self {
            op,
            session_id: session_id.into(),
            user_id,
            queued_at: Instant::now(),
        }
    }
}

/// An immutable, ordered batch of operations ready for transmission.
#[derive(Debug)]
pub struct OpBatch {
    /// Operations in submission order.
    pub ops: Vec<BatchedOp>,
    /// When this batch was created.
    pub created_at: Instant,
    /// Sequence number (monotonically increasing per `OpBatcher`).
    pub seq: u64,
}

impl OpBatch {
    /// Number of operations in this batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Return `true` when the batch contains no operations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Rough byte size estimate: 64 bytes overhead per op plus path length.
    #[must_use]
    pub fn estimated_bytes(&self) -> usize {
        self.ops
            .iter()
            .map(|b| 64 + b.op.path.len())
            .sum()
    }
}

/// Configuration for [`OpBatcher`].
#[derive(Debug, Clone)]
pub struct BatcherConfig {
    /// Maximum number of operations per batch.  When this is reached the
    /// pending buffer is flushed immediately.
    pub max_ops: usize,
    /// Maximum estimated payload size in bytes.  Flush when exceeded.
    pub max_bytes: usize,
    /// Maximum time to wait before flushing an incomplete batch.
    pub max_age: Duration,
    /// Whether to enable *op coalescing*: consecutive `Update` operations on
    /// the same path by the same user are merged into a single op.
    pub coalesce_updates: bool,
}

impl Default for BatcherConfig {
    fn default() -> Self {
        Self {
            max_ops: 64,
            max_bytes: 16 * 1024,   // 16 KiB
            max_age: Duration::from_millis(50),
            coalesce_updates: true,
        }
    }
}

/// Aggregate statistics across all batches produced by an [`OpBatcher`].
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    /// Total batches flushed.
    pub batches_flushed: u64,
    /// Total individual operations processed.
    pub ops_processed: u64,
    /// Total ops that were coalesced away (duplicate updates).
    pub ops_coalesced: u64,
    /// Maximum batch size observed (op count).
    pub max_batch_size: usize,
    /// Minimum batch size observed (op count).
    pub min_batch_size: usize,
}

impl BatchStats {
    fn record_flush(&mut self, batch_size: usize, coalesced: u64) {
        self.batches_flushed += 1;
        self.ops_processed += batch_size as u64;
        self.ops_coalesced += coalesced;
        if batch_size > self.max_batch_size {
            self.max_batch_size = batch_size;
        }
        if self.batches_flushed == 1 || batch_size < self.min_batch_size {
            self.min_batch_size = batch_size;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OpBatcher
// ─────────────────────────────────────────────────────────────────────────────

/// Accumulates operations and flushes them as [`OpBatch`]es based on size
/// and age constraints.
///
/// The batcher is intentionally **synchronous** and lock-free — the caller is
/// responsible for invoking [`Self::push`] and [`Self::flush_if_ready`] from
/// a single thread (or behind an external mutex), in line with the rest of the
/// collaboration engine's actor model.
pub struct OpBatcher {
    config: BatcherConfig,
    pending: VecDeque<BatchedOp>,
    /// Estimated byte size of `pending`.
    pending_bytes: usize,
    /// Wall-clock time when the first op in `pending` was enqueued.
    batch_start: Option<Instant>,
    /// Monotonic sequence counter.
    seq: u64,
    /// Coalesced-op counter for the current batch.
    coalesced_current: u64,
    /// Aggregate statistics.
    pub stats: BatchStats,
    /// Completed batches awaiting consumption.
    ready: VecDeque<OpBatch>,
}

impl OpBatcher {
    /// Create a new batcher with the given configuration.
    pub fn new(config: BatcherConfig) -> Self {
        Self {
            config,
            pending: VecDeque::new(),
            pending_bytes: 0,
            batch_start: None,
            seq: 0,
            coalesced_current: 0,
            stats: BatchStats::default(),
            ready: VecDeque::new(),
        }
    }

    /// Create a batcher with default configuration.
    pub fn default_config() -> Self {
        Self::new(BatcherConfig::default())
    }

    /// Push an operation into the pending buffer.
    ///
    /// If coalescing is enabled and the operation is an `Update` on the same
    /// path/user as the most recent pending op, the two are merged in place
    /// (the old value of the earlier op is preserved; the new value is updated
    /// to the incoming one).
    ///
    /// After pushing, the method checks whether the batch should be flushed
    /// and appends to `ready` if so.
    pub fn push(&mut self, bop: BatchedOp) {
        // Attempt coalescing.
        if self.config.coalesce_updates {
            if let Some(last) = self.pending.back_mut() {
                if last.user_id == bop.user_id
                    && last.op.path == bop.op.path
                    && last.session_id == bop.session_id
                {
                    if let (
                        OpType::Update {
                            new_value: ref mut last_new,
                            ..
                        },
                        OpType::Update {
                            new_value: incoming_new,
                            ..
                        },
                    ) = (&mut last.op.op_type, &bop.op.op_type)
                    {
                        *last_new = *incoming_new;
                        self.coalesced_current += 1;
                        return;
                    }
                }
            }
        }

        // Record when the first op in a batch arrived.
        if self.pending.is_empty() {
            self.batch_start = Some(Instant::now());
        }

        let byte_estimate = 64 + bop.op.path.len();
        self.pending_bytes += byte_estimate;
        self.pending.push_back(bop);

        // Check flush conditions.
        if self.should_flush() {
            self.do_flush();
        }
    }

    /// Force a flush of all pending operations into a new [`OpBatch`],
    /// regardless of size/age constraints.  Returns `true` if any ops were
    /// flushed.
    pub fn flush(&mut self) -> bool {
        if self.pending.is_empty() {
            return false;
        }
        self.do_flush();
        true
    }

    /// Flush the pending buffer if the size or age threshold is exceeded.
    ///
    /// Returns `true` if a batch was emitted.
    pub fn flush_if_ready(&mut self) -> bool {
        if self.should_flush() {
            self.do_flush();
            true
        } else {
            false
        }
    }

    /// Return all completed batches that have not yet been consumed, draining
    /// the internal ready queue.
    pub fn drain_ready(&mut self) -> Vec<OpBatch> {
        self.ready.drain(..).collect()
    }

    /// Peek at the number of ready batches without consuming them.
    #[must_use]
    pub fn ready_count(&self) -> usize {
        self.ready.len()
    }

    /// Number of operations currently in the pending (unflushed) buffer.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Estimated byte size of the pending buffer.
    #[must_use]
    pub fn pending_bytes(&self) -> usize {
        self.pending_bytes
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn should_flush(&self) -> bool {
        if self.pending.is_empty() {
            return false;
        }
        if self.pending.len() >= self.config.max_ops {
            return true;
        }
        if self.pending_bytes >= self.config.max_bytes {
            return true;
        }
        if let Some(start) = self.batch_start {
            if start.elapsed() >= self.config.max_age {
                return true;
            }
        }
        false
    }

    fn do_flush(&mut self) {
        let ops: Vec<BatchedOp> = self.pending.drain(..).collect();
        let batch_size = ops.len();
        let coalesced = self.coalesced_current;
        self.coalesced_current = 0;
        self.pending_bytes = 0;
        self.batch_start = None;

        self.seq += 1;
        let batch = OpBatch {
            ops,
            created_at: Instant::now(),
            seq: self.seq,
        };

        self.stats.record_flush(batch_size, coalesced);
        self.ready.push_back(batch);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SyncBatchBridge — wires OpBatcher into the sync pipeline
// ─────────────────────────────────────────────────────────────────────────────

/// A complete batch of operations prepared for network transmission.
///
/// This is the "over-the-wire" envelope produced by [`SyncEngine`] when
/// it drains its internal [`OpBatcher`].
#[derive(Debug, Clone)]
pub struct BatchedOps {
    /// Operations in submission order.
    pub ops: Vec<Operation>,
    /// Session-scoped user identifier.
    pub sender_id: String,
    /// `(min_seq, max_seq)` inclusive range of operation IDs in this batch.
    pub seq_range: (u64, u64),
}

/// A synchronisation engine that accumulates operations via an [`OpBatcher`]
/// before forwarding them as [`BatchedOps`] messages.
///
/// # Example
///
/// ```rust
/// use oximedia_collab::op_batcher::{SyncEngine, BatcherConfig};
/// let mut engine = SyncEngine::new("session-1", "user-42");
/// // Operations are now buffered and sent as batches.
/// ```
pub struct SyncEngine {
    /// Session identifier (passed through to all batched messages).
    session_id: String,
    /// User identifier of the local peer.
    user_id: String,
    /// Inner batcher that enforces size/age policies.
    batcher: OpBatcher,
    /// Queue of batched message envelopes ready for network dispatch.
    outbox: VecDeque<BatchedOps>,
}

impl SyncEngine {
    /// Create a new sync engine with default batching configuration.
    pub fn new(session_id: impl Into<String>, user_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            user_id: user_id.into(),
            batcher: OpBatcher::default_config(),
            outbox: VecDeque::new(),
        }
    }

    /// Create a sync engine with explicit batching parameters.
    ///
    /// * `window_ms` — maximum age of a batch before it is force-flushed.
    /// * `max_size` — maximum number of operations in a single batch.
    pub fn with_batcher(
        session_id: impl Into<String>,
        user_id: impl Into<String>,
        window_ms: u64,
        max_size: usize,
    ) -> Self {
        let config = BatcherConfig {
            max_ops: max_size,
            max_age: Duration::from_millis(window_ms),
            ..BatcherConfig::default()
        };
        Self {
            session_id: session_id.into(),
            user_id: user_id.into(),
            batcher: OpBatcher::new(config),
            outbox: VecDeque::new(),
        }
    }

    /// Submit an operation to the batching buffer.
    ///
    /// The engine automatically flushes to `outbox` when the configured size
    /// or age threshold is exceeded.
    pub fn push_operation(&mut self, op: Operation) {
        let user_id_num: u32 = {
            // Stable hash of the user_id string → u32 for BatchedOp.
            let mut h: u32 = 2166136261;
            for b in self.user_id.bytes() {
                h = h.wrapping_mul(16777619).wrapping_add(u32::from(b));
            }
            h
        };
        let bop = BatchedOp::new(op, &self.session_id, user_id_num);
        self.batcher.push(bop);
        self.drain_batcher();
    }

    /// Force a flush of any pending operations regardless of thresholds.
    pub fn flush(&mut self) {
        self.batcher.flush();
        self.drain_batcher();
    }

    /// Check age threshold and flush if the oldest pending op is stale.
    pub fn flush_if_ready(&mut self) {
        self.batcher.flush_if_ready();
        self.drain_batcher();
    }

    /// Drain all ready [`BatchedOps`] from the outbox.
    ///
    /// Call this after `push_operation` or `flush` to retrieve messages ready
    /// for transmission.
    pub fn drain_outbox(&mut self) -> Vec<BatchedOps> {
        self.outbox.drain(..).collect()
    }

    /// Number of complete batches waiting in the outbox.
    pub fn outbox_count(&self) -> usize {
        self.outbox.len()
    }

    /// Number of operations currently buffered (not yet flushed).
    pub fn pending_count(&self) -> usize {
        self.batcher.pending_count()
    }

    /// Aggregate statistics from the underlying [`OpBatcher`].
    pub fn stats(&self) -> &BatchStats {
        &self.batcher.stats
    }

    // ── Internal helpers ───────────────────────────────────────────────────────

    fn drain_batcher(&mut self) {
        for batch in self.batcher.drain_ready() {
            if batch.is_empty() {
                continue;
            }
            let ops: Vec<Operation> = batch.ops.iter().map(|b| b.op.clone()).collect();
            let min_id = ops.iter().map(|o| o.id).min().unwrap_or(0);
            let max_id = ops.iter().map(|o| o.id).max().unwrap_or(0);
            self.outbox.push_back(BatchedOps {
                ops,
                sender_id: self.user_id.clone(),
                seq_range: (min_id, max_id),
            });
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operation_log::{OpType, Operation};

    fn make_update_op(id: u64, path: &str, old: f32, new_value: f32) -> Operation {
        Operation::new(
            id,
            1,
            0,
            path,
            OpType::Update {
                index: 0,
                old,
                new_value,
            },
        )
    }

    fn make_insert_op(id: u64, path: &str) -> Operation {
        Operation::new(id, 1, 0, path, OpType::Insert { index: 0, value: 0.0 })
    }

    fn bop(op: Operation) -> BatchedOp {
        BatchedOp::new(op, "sess-1", 1)
    }

    // ── Basic push and flush ──────────────────────────────────────────────────

    #[test]
    fn test_push_and_explicit_flush() {
        let mut batcher = OpBatcher::new(BatcherConfig {
            max_ops: 100,
            ..Default::default()
        });
        batcher.push(bop(make_insert_op(1, "track/1")));
        batcher.push(bop(make_insert_op(2, "track/2")));
        assert_eq!(batcher.pending_count(), 2);

        let flushed = batcher.flush();
        assert!(flushed);
        assert_eq!(batcher.pending_count(), 0);
        assert_eq!(batcher.ready_count(), 1);
    }

    #[test]
    fn test_flush_empty_returns_false() {
        let mut batcher = OpBatcher::default_config();
        assert!(!batcher.flush());
    }

    // ── Size-triggered flush ──────────────────────────────────────────────────

    #[test]
    fn test_flush_on_max_ops() {
        let config = BatcherConfig {
            max_ops: 3,
            coalesce_updates: false,
            ..Default::default()
        };
        let mut batcher = OpBatcher::new(config);
        batcher.push(bop(make_insert_op(1, "a")));
        batcher.push(bop(make_insert_op(2, "b")));
        assert_eq!(batcher.ready_count(), 0); // not yet
        batcher.push(bop(make_insert_op(3, "c"))); // triggers flush
        assert_eq!(batcher.ready_count(), 1);
        assert_eq!(batcher.pending_count(), 0);
    }

    // ── Coalescing ────────────────────────────────────────────────────────────

    #[test]
    fn test_coalesce_consecutive_updates_same_path() {
        let config = BatcherConfig {
            max_ops: 100,
            coalesce_updates: true,
            ..Default::default()
        };
        let mut batcher = OpBatcher::new(config);
        batcher.push(bop(make_update_op(1, "gain", 0.0, 0.5)));
        batcher.push(bop(make_update_op(2, "gain", 0.5, 0.8)));
        batcher.push(bop(make_update_op(3, "gain", 0.8, 1.0)));

        // All three ops on same path should coalesce into 1.
        assert_eq!(batcher.pending_count(), 1);
        batcher.flush();
        let batches = batcher.drain_ready();
        assert_eq!(batches[0].len(), 1);

        // The surviving op should carry the latest new_value (1.0).
        if let OpType::Update { new_value, .. } = batches[0].ops[0].op.op_type {
            assert!((new_value - 1.0).abs() < 1e-6);
        } else {
            panic!("expected Update op");
        }
    }

    #[test]
    fn test_no_coalesce_different_paths() {
        let config = BatcherConfig {
            max_ops: 100,
            coalesce_updates: true,
            ..Default::default()
        };
        let mut batcher = OpBatcher::new(config);
        batcher.push(bop(make_update_op(1, "gain", 0.0, 0.5)));
        batcher.push(bop(make_update_op(2, "pan", 0.0, 0.3)));
        // Different paths — no coalescing.
        assert_eq!(batcher.pending_count(), 2);
    }

    #[test]
    fn test_coalescing_disabled() {
        let config = BatcherConfig {
            max_ops: 100,
            coalesce_updates: false,
            ..Default::default()
        };
        let mut batcher = OpBatcher::new(config);
        batcher.push(bop(make_update_op(1, "gain", 0.0, 0.5)));
        batcher.push(bop(make_update_op(2, "gain", 0.5, 0.8)));
        assert_eq!(batcher.pending_count(), 2);
    }

    // ── Drain ready ───────────────────────────────────────────────────────────

    #[test]
    fn test_drain_ready_consumes_all_batches() {
        let config = BatcherConfig {
            max_ops: 2,
            coalesce_updates: false,
            ..Default::default()
        };
        let mut batcher = OpBatcher::new(config);
        // First batch: ops 1, 2 → auto-flushed
        batcher.push(bop(make_insert_op(1, "a")));
        batcher.push(bop(make_insert_op(2, "b")));
        // Second batch: op 3 (explicit flush)
        batcher.push(bop(make_insert_op(3, "c")));
        batcher.flush();

        let batches = batcher.drain_ready();
        assert_eq!(batches.len(), 2);
        assert_eq!(batcher.ready_count(), 0); // drained
    }

    // ── Stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_incremented_on_flush() {
        let mut batcher = OpBatcher::default_config();
        batcher.push(bop(make_insert_op(1, "a")));
        batcher.push(bop(make_insert_op(2, "b")));
        batcher.flush();

        assert_eq!(batcher.stats.batches_flushed, 1);
        assert_eq!(batcher.stats.ops_processed, 2);
    }

    #[test]
    fn test_stats_coalesced_count() {
        let config = BatcherConfig {
            max_ops: 100,
            coalesce_updates: true,
            ..Default::default()
        };
        let mut batcher = OpBatcher::new(config);
        batcher.push(bop(make_update_op(1, "vol", 0.0, 0.5)));
        batcher.push(bop(make_update_op(2, "vol", 0.5, 0.9)));
        batcher.flush();

        // One op coalesced away.
        assert_eq!(batcher.stats.ops_coalesced, 1);
    }

    // ── Batch metadata ────────────────────────────────────────────────────────

    #[test]
    fn test_batch_sequence_numbers_monotone() {
        let config = BatcherConfig {
            max_ops: 1,
            coalesce_updates: false,
            ..Default::default()
        };
        let mut batcher = OpBatcher::new(config);
        batcher.push(bop(make_insert_op(1, "a")));
        batcher.push(bop(make_insert_op(2, "b")));
        let batches = batcher.drain_ready();
        assert_eq!(batches.len(), 2);
        assert!(batches[0].seq < batches[1].seq);
    }

    #[test]
    fn test_estimated_bytes_non_zero() {
        let mut batcher = OpBatcher::default_config();
        batcher.push(bop(make_insert_op(1, "audio/track/0")));
        assert!(batcher.pending_bytes() > 0);
    }

    #[test]
    fn test_batch_estimated_bytes() {
        let config = BatcherConfig {
            max_ops: 100,
            coalesce_updates: false,
            ..Default::default()
        };
        let mut batcher = OpBatcher::new(config);
        batcher.push(bop(make_insert_op(1, "audio/track/0")));
        batcher.push(bop(make_insert_op(2, "audio/track/1")));
        batcher.flush();
        let batches = batcher.drain_ready();
        assert!(batches[0].estimated_bytes() > 0);
    }

    // ── SyncEngine ────────────────────────────────────────────────────────────

    fn make_op(id: u64) -> Operation {
        make_insert_op(id, &format!("track/{id}"))
    }

    #[test]
    fn test_sync_engine_single_op_flushed_explicitly() {
        let mut engine = SyncEngine::new("session-1", "user-1");
        engine.push_operation(make_op(1));
        engine.flush();
        let batches = engine.drain_outbox();
        assert_eq!(batches.len(), 1, "After explicit flush, one batch in outbox");
        assert_eq!(batches[0].ops.len(), 1);
        assert_eq!(batches[0].sender_id, "user-1");
    }

    #[test]
    fn test_sync_engine_ops_within_window_batched() {
        // window = 1000ms, max_size = 100 — all 5 ops arrive before flush.
        let mut engine = SyncEngine::with_batcher("sess", "user", 1_000, 100);
        for i in 1..=5 {
            engine.push_operation(make_op(i));
        }
        assert_eq!(engine.outbox_count(), 0, "No auto-flush yet");
        engine.flush();
        let batches = engine.drain_outbox();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].ops.len(), 5);
    }

    #[test]
    fn test_sync_engine_max_size_triggers_early_flush() {
        // max_size = 3 — after every 3 ops, a batch should flush.
        let mut engine = SyncEngine::with_batcher("sess", "user", 60_000, 3);
        for i in 1..=6u64 {
            engine.push_operation(make_op(i));
        }
        let batches = engine.drain_outbox();
        assert_eq!(batches.len(), 2, "6 ops at max_size=3 → 2 batches");
    }

    #[test]
    fn test_sync_engine_ordering_preserved() {
        let mut engine = SyncEngine::with_batcher("sess", "user", 60_000, 10);
        for i in 1..=4u64 {
            engine.push_operation(make_op(i));
        }
        engine.flush();
        let batches = engine.drain_outbox();
        let ids: Vec<u64> = batches.iter().flat_map(|b| b.ops.iter().map(|o| o.id)).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(ids, sorted, "Operation order should be preserved in batch");
    }

    #[test]
    fn test_sync_engine_seq_range_correct() {
        let mut engine = SyncEngine::with_batcher("sess", "user", 60_000, 100);
        engine.push_operation(make_op(5));
        engine.push_operation(make_op(10));
        engine.push_operation(make_op(3));
        engine.flush();
        let batches = engine.drain_outbox();
        let (min_id, max_id) = batches[0].seq_range;
        assert_eq!(min_id, 3, "min seq should be 3");
        assert_eq!(max_id, 10, "max seq should be 10");
    }

    #[test]
    fn test_sync_engine_stats_track_flushes() {
        let mut engine = SyncEngine::new("sess", "user");
        engine.push_operation(make_op(1));
        engine.flush();
        assert_eq!(engine.stats().batches_flushed, 1);
    }

    #[test]
    fn test_sync_engine_drain_outbox_clears() {
        let mut engine = SyncEngine::new("sess", "user");
        engine.push_operation(make_op(1));
        engine.flush();
        assert_eq!(engine.outbox_count(), 1);
        let _ = engine.drain_outbox();
        assert_eq!(engine.outbox_count(), 0, "Drain should clear the outbox");
    }

    #[test]
    fn test_sync_engine_empty_flush_no_batch() {
        let mut engine = SyncEngine::new("sess", "user");
        engine.flush(); // nothing pending
        assert_eq!(engine.outbox_count(), 0, "Empty flush should not create a batch");
    }
}
