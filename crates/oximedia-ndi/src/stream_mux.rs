// #![allow(dead_code)]
//! NDI stream multiplexer for `oximedia-ndi`.
//!
//! Multiplexes frames from multiple NDI sources into a single ordered output
//! stream. Supports configurable mixing strategies:
//!
//! - **Primary** — output only from the primary (first) active source; fall
//!   back to the next available source when the primary is silent.
//! - **RoundRobin** — cycle through sources in order, emitting one frame per
//!   source per pass.
//! - **Merge** — emit frames from *all* sources in timestamp order.
//!
//! Sources are identified by a `u64` slot ID and can be added or removed at
//! any time. Each source slot has a configurable priority and an independent
//! frame queue.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]

use std::collections::{BTreeMap, VecDeque};

// ---------------------------------------------------------------------------
// MuxStrategy
// ---------------------------------------------------------------------------

/// How the multiplexer selects frames from its source slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MuxStrategy {
    /// Emit from the highest-priority active source; fall back automatically.
    Primary,
    /// Cycle through active sources in priority order, one frame each.
    RoundRobin,
    /// Emit all queued frames from all sources sorted by timestamp.
    Merge,
}

impl Default for MuxStrategy {
    fn default() -> Self {
        Self::Primary
    }
}

// ---------------------------------------------------------------------------
// MuxedFrame
// ---------------------------------------------------------------------------

/// A frame emitted by the multiplexer, annotated with its source slot.
#[derive(Debug, Clone)]
pub struct MuxedFrame {
    /// The slot ID of the source this frame came from.
    pub source_id: u64,
    /// Raw frame data.
    pub data: Vec<u8>,
    /// Presentation timestamp in microseconds.
    pub timestamp_us: u64,
    /// The priority of the source slot at the time of emission.
    pub priority: u32,
}

// ---------------------------------------------------------------------------
// SourceSlot
// ---------------------------------------------------------------------------

/// Internal state for one NDI source input.
#[derive(Debug)]
struct SourceSlot {
    id: u64,
    priority: u32,
    active: bool,
    queue: VecDeque<(Vec<u8>, u64)>, // (data, timestamp_us)
    max_queue_depth: usize,
    frames_pushed: u64,
    frames_dropped: u64,
    frames_emitted: u64,
}

impl SourceSlot {
    fn new(id: u64, priority: u32, max_queue_depth: usize) -> Self {
        Self {
            id,
            priority,
            active: true,
            queue: VecDeque::with_capacity(max_queue_depth),
            max_queue_depth,
            frames_pushed: 0,
            frames_dropped: 0,
            frames_emitted: 0,
        }
    }

    fn push(&mut self, data: Vec<u8>, timestamp_us: u64) -> bool {
        if self.queue.len() >= self.max_queue_depth {
            self.frames_dropped += 1;
            return false;
        }
        self.queue.push_back((data, timestamp_us));
        self.frames_pushed += 1;
        true
    }

    fn pop(&mut self) -> Option<(Vec<u8>, u64)> {
        let item = self.queue.pop_front()?;
        self.frames_emitted += 1;
        Some(item)
    }

    fn is_ready(&self) -> bool {
        self.active && !self.queue.is_empty()
    }
}

// ---------------------------------------------------------------------------
// SlotStats
// ---------------------------------------------------------------------------

/// Per-slot statistics exposed to callers.
#[derive(Debug, Clone)]
pub struct SlotStats {
    /// Slot identifier.
    pub id: u64,
    /// Source priority.
    pub priority: u32,
    /// Whether the slot is active.
    pub active: bool,
    /// Current queue depth.
    pub queue_depth: usize,
    /// Total frames pushed.
    pub frames_pushed: u64,
    /// Total frames dropped due to overflow.
    pub frames_dropped: u64,
    /// Total frames emitted.
    pub frames_emitted: u64,
}

// ---------------------------------------------------------------------------
// MuxStats
// ---------------------------------------------------------------------------

/// Aggregate multiplexer statistics.
#[derive(Debug, Clone, Default)]
pub struct MuxStats {
    /// Number of currently registered source slots.
    pub source_count: usize,
    /// Number of active source slots.
    pub active_sources: usize,
    /// Total frames emitted across all sources.
    pub total_emitted: u64,
    /// Total frames dropped across all sources.
    pub total_dropped: u64,
    /// Current round-robin cursor.
    pub rr_cursor: usize,
}

// ---------------------------------------------------------------------------
// StreamMux
// ---------------------------------------------------------------------------

/// Multiplexes frames from multiple NDI source slots into a single output
/// stream.
#[derive(Debug)]
pub struct StreamMux {
    strategy: MuxStrategy,
    /// Slots keyed by slot ID, ordered by insertion.
    slots: BTreeMap<u64, SourceSlot>,
    /// Per-frame queue depth limit for new slots.
    default_queue_depth: usize,
    /// Next ID to assign.
    next_id: u64,
    /// Round-robin cursor (index into sorted priority list).
    rr_cursor: usize,
    /// Total frames emitted by this mux.
    total_emitted: u64,
    /// Total frames dropped by this mux (across all slots).
    total_dropped: u64,
}

impl StreamMux {
    /// Create a new multiplexer with the given strategy and default per-source
    /// queue depth.
    pub fn new(strategy: MuxStrategy, default_queue_depth: usize) -> Self {
        Self {
            strategy,
            slots: BTreeMap::new(),
            default_queue_depth: default_queue_depth.max(1),
            next_id: 1,
            rr_cursor: 0,
            total_emitted: 0,
            total_dropped: 0,
        }
    }

    /// Add a new source slot with the given priority. Returns the slot ID.
    ///
    /// Higher priority values take precedence in `Primary` mode.
    pub fn add_source(&mut self, priority: u32) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let slot = SourceSlot::new(id, priority, self.default_queue_depth);
        self.slots.insert(id, slot);
        id
    }

    /// Add a source with a custom queue depth.
    pub fn add_source_with_depth(&mut self, priority: u32, queue_depth: usize) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let slot = SourceSlot::new(id, priority, queue_depth.max(1));
        self.slots.insert(id, slot);
        id
    }

    /// Remove a source slot by ID. Returns `true` if it existed.
    pub fn remove_source(&mut self, id: u64) -> bool {
        self.slots.remove(&id).is_some()
    }

    /// Set a slot's active state. Inactive slots are skipped during emission.
    pub fn set_active(&mut self, id: u64, active: bool) -> bool {
        if let Some(slot) = self.slots.get_mut(&id) {
            slot.active = active;
            true
        } else {
            false
        }
    }

    /// Push a frame into the given source slot. Returns `false` if the slot
    /// does not exist or its queue is full.
    pub fn push(&mut self, source_id: u64, data: Vec<u8>, timestamp_us: u64) -> bool {
        match self.slots.get_mut(&source_id) {
            Some(slot) => {
                let ok = slot.push(data, timestamp_us);
                if !ok {
                    self.total_dropped += 1;
                }
                ok
            }
            None => false,
        }
    }

    /// Pull the next output frame according to the configured strategy.
    /// Returns `None` when no frames are available.
    pub fn pull(&mut self) -> Option<MuxedFrame> {
        match self.strategy {
            MuxStrategy::Primary => self.pull_primary(),
            MuxStrategy::RoundRobin => self.pull_round_robin(),
            MuxStrategy::Merge => self.pull_merge(),
        }
    }

    /// Pull all available output frames.
    pub fn pull_all(&mut self) -> Vec<MuxedFrame> {
        let mut out = Vec::new();
        while let Some(frame) = self.pull() {
            out.push(frame);
            // Guard against infinite loop if strategy produces synthetic frames.
            if out.len() > 65536 {
                break;
            }
        }
        out
    }

    fn pull_primary(&mut self) -> Option<MuxedFrame> {
        // Find the highest-priority ready slot.
        let best_id = self
            .slots
            .values()
            .filter(|s| s.is_ready())
            .max_by_key(|s| s.priority)?
            .id;

        self.emit(best_id)
    }

    fn pull_round_robin(&mut self) -> Option<MuxedFrame> {
        // Collect ready slot IDs sorted by priority desc, then id asc.
        let mut ready: Vec<u64> = self
            .slots
            .values()
            .filter(|s| s.is_ready())
            .map(|s| s.id)
            .collect();

        if ready.is_empty() {
            return None;
        }

        // Sort by (priority desc, id asc) for stable round-robin.
        ready.sort_by(|&a, &b| {
            let pa = self.slots[&a].priority;
            let pb = self.slots[&b].priority;
            pb.cmp(&pa).then(a.cmp(&b))
        });

        let idx = self.rr_cursor % ready.len();
        self.rr_cursor = self.rr_cursor.wrapping_add(1);
        self.emit(ready[idx])
    }

    fn pull_merge(&mut self) -> Option<MuxedFrame> {
        // Pick the slot whose front frame has the earliest timestamp.
        let best_id = self
            .slots
            .values()
            .filter(|s| s.is_ready())
            .min_by_key(|s| s.queue.front().map(|(_, ts)| *ts).unwrap_or(u64::MAX))?
            .id;

        self.emit(best_id)
    }

    fn emit(&mut self, source_id: u64) -> Option<MuxedFrame> {
        let slot = self.slots.get_mut(&source_id)?;
        let (data, timestamp_us) = slot.pop()?;
        let priority = slot.priority;
        self.total_emitted += 1;
        Some(MuxedFrame {
            source_id,
            data,
            timestamp_us,
            priority,
        })
    }

    /// Return the number of registered source slots.
    pub fn source_count(&self) -> usize {
        self.slots.len()
    }

    /// Return the number of active source slots.
    pub fn active_source_count(&self) -> usize {
        self.slots.values().filter(|s| s.active).count()
    }

    /// Return aggregate statistics.
    pub fn stats(&self) -> MuxStats {
        MuxStats {
            source_count: self.slots.len(),
            active_sources: self.slots.values().filter(|s| s.active).count(),
            total_emitted: self.total_emitted,
            total_dropped: self.total_dropped,
            rr_cursor: self.rr_cursor,
        }
    }

    /// Return per-slot statistics.
    pub fn slot_stats(&self, id: u64) -> Option<SlotStats> {
        let slot = self.slots.get(&id)?;
        Some(SlotStats {
            id: slot.id,
            priority: slot.priority,
            active: slot.active,
            queue_depth: slot.queue.len(),
            frames_pushed: slot.frames_pushed,
            frames_dropped: slot.frames_dropped,
            frames_emitted: slot.frames_emitted,
        })
    }

    /// Return the strategy in use.
    pub fn strategy(&self) -> MuxStrategy {
        self.strategy
    }

    /// Change the mixing strategy at runtime.
    pub fn set_strategy(&mut self, strategy: MuxStrategy) {
        self.strategy = strategy;
        self.rr_cursor = 0;
    }

    /// Flush and discard all buffered frames across all slots.
    pub fn flush_all(&mut self) {
        for slot in self.slots.values_mut() {
            slot.queue.clear();
        }
    }

    /// Return the total number of frames emitted.
    pub fn total_emitted(&self) -> u64 {
        self.total_emitted
    }

    /// Return the total number of frames dropped due to queue overflow.
    pub fn total_dropped(&self) -> u64 {
        self.total_dropped
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(value: u8, ts: u64) -> (Vec<u8>, u64) {
        (vec![value; 4], ts)
    }

    #[test]
    fn test_add_remove_source() {
        let mut mux = StreamMux::new(MuxStrategy::Primary, 8);
        let id = mux.add_source(10);
        assert_eq!(mux.source_count(), 1);
        assert!(mux.remove_source(id));
        assert_eq!(mux.source_count(), 0);
    }

    #[test]
    fn test_push_returns_false_for_unknown_id() {
        let mut mux = StreamMux::new(MuxStrategy::Primary, 8);
        assert!(!mux.push(999, vec![0; 4], 0));
    }

    #[test]
    fn test_primary_picks_highest_priority() {
        let mut mux = StreamMux::new(MuxStrategy::Primary, 8);
        let low = mux.add_source(1);
        let high = mux.add_source(10);

        let (d, ts) = frame(1, 0);
        mux.push(low, d, ts);
        let (d, ts) = frame(2, 0);
        mux.push(high, d, ts);

        let out = mux.pull().expect("should emit");
        assert_eq!(out.source_id, high, "should pick highest priority");
    }

    #[test]
    fn test_primary_falls_back_when_primary_empty() {
        let mut mux = StreamMux::new(MuxStrategy::Primary, 8);
        let low = mux.add_source(1);
        let _high = mux.add_source(10); // high is empty

        let (d, ts) = frame(1, 0);
        mux.push(low, d, ts);

        let out = mux.pull().expect("fallback to low");
        assert_eq!(out.source_id, low);
    }

    #[test]
    fn test_pull_returns_none_when_all_empty() {
        let mut mux = StreamMux::new(MuxStrategy::Primary, 8);
        let _id = mux.add_source(5);
        assert!(mux.pull().is_none());
    }

    #[test]
    fn test_round_robin_cycles() {
        let mut mux = StreamMux::new(MuxStrategy::RoundRobin, 8);
        let id1 = mux.add_source(5);
        let id2 = mux.add_source(5);

        // Push 2 frames into each
        for ts in 0..2u64 {
            let (d, t) = frame(1, ts * 33333);
            mux.push(id1, d, t);
            let (d, t) = frame(2, ts * 33333 + 1);
            mux.push(id2, d, t);
        }

        let f1 = mux.pull().expect("first");
        let f2 = mux.pull().expect("second");
        // Should have picked from different sources
        assert_ne!(f1.source_id, f2.source_id, "round-robin should alternate");
    }

    #[test]
    fn test_merge_picks_earliest_timestamp() {
        let mut mux = StreamMux::new(MuxStrategy::Merge, 8);
        let id1 = mux.add_source(5);
        let id2 = mux.add_source(5);

        mux.push(id1, vec![0; 4], 1000);
        mux.push(id2, vec![0; 4], 500); // earlier

        let out = mux.pull().expect("should emit");
        assert_eq!(out.source_id, id2, "merge should pick earliest timestamp");
    }

    #[test]
    fn test_inactive_source_skipped() {
        let mut mux = StreamMux::new(MuxStrategy::Primary, 8);
        let active = mux.add_source(5);
        let inactive = mux.add_source(10);
        mux.set_active(inactive, false);

        mux.push(inactive, vec![0; 4], 0);
        mux.push(active, vec![1; 4], 0);

        let out = mux.pull().expect("should emit");
        assert_eq!(out.source_id, active);
    }

    #[test]
    fn test_queue_overflow_drops() {
        let mut mux = StreamMux::new(MuxStrategy::Primary, 2);
        let id = mux.add_source(5);

        mux.push(id, vec![0; 4], 0);
        mux.push(id, vec![0; 4], 1);
        let ok = mux.push(id, vec![0; 4], 2); // should fail
        assert!(!ok);
        assert!(mux.total_dropped() > 0);
    }

    #[test]
    fn test_pull_all() {
        let mut mux = StreamMux::new(MuxStrategy::Merge, 8);
        let id = mux.add_source(5);
        for ts in 0..5u64 {
            mux.push(id, vec![ts as u8; 4], ts * 1000);
        }
        let all = mux.pull_all();
        assert_eq!(all.len(), 5);
    }

    #[test]
    fn test_stats() {
        let mut mux = StreamMux::new(MuxStrategy::Primary, 8);
        let id = mux.add_source(5);
        mux.push(id, vec![0; 4], 0);
        mux.pull();

        let s = mux.stats();
        assert_eq!(s.source_count, 1);
        assert_eq!(s.total_emitted, 1);
    }

    #[test]
    fn test_slot_stats() {
        let mut mux = StreamMux::new(MuxStrategy::Primary, 8);
        let id = mux.add_source(7);
        mux.push(id, vec![0; 4], 0);
        let ss = mux.slot_stats(id).expect("slot exists");
        assert_eq!(ss.id, id);
        assert_eq!(ss.frames_pushed, 1);
        assert_eq!(ss.priority, 7);
    }

    #[test]
    fn test_set_strategy_runtime() {
        let mut mux = StreamMux::new(MuxStrategy::Primary, 8);
        assert_eq!(mux.strategy(), MuxStrategy::Primary);
        mux.set_strategy(MuxStrategy::RoundRobin);
        assert_eq!(mux.strategy(), MuxStrategy::RoundRobin);
    }

    #[test]
    fn test_flush_all_clears_queues() {
        let mut mux = StreamMux::new(MuxStrategy::Primary, 8);
        let id = mux.add_source(5);
        mux.push(id, vec![0; 4], 0);
        mux.flush_all();
        assert!(mux.pull().is_none());
    }

    #[test]
    fn test_muxed_frame_priority_field() {
        let mut mux = StreamMux::new(MuxStrategy::Primary, 8);
        let id = mux.add_source(42);
        mux.push(id, vec![0; 4], 0);
        let frame = mux.pull().expect("frame");
        assert_eq!(frame.priority, 42);
    }

    #[test]
    fn test_add_source_with_depth() {
        let mut mux = StreamMux::new(MuxStrategy::Primary, 4);
        let id = mux.add_source_with_depth(5, 16);
        for i in 0..16u64 {
            assert!(mux.push(id, vec![0; 4], i * 1000), "push {i} should succeed");
        }
        // 17th push should fail
        assert!(!mux.push(id, vec![0; 4], 17000));
    }
}
