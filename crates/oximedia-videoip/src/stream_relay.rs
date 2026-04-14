//! Stream relay: re-broadcasts received streams to multiple downstream receivers.
//!
//! A `StreamRelay` accepts frames from one or more upstream sources and fans them
//! out to a configurable set of downstream sinks.  The relay is protocol-agnostic:
//! callers push raw frame data in and register destination handles that receive copies.
//!
//! # Design
//!
//! ```text
//! ┌──────────┐         ┌──────────────┐        ┌──────────────┐
//! │ Source A │──push──▶│  StreamRelay │──fan──▶│ Downstream 1 │
//! │ Source B │──push──▶│              │──out──▶│ Downstream 2 │
//! └──────────┘         └──────────────┘        └──────────────┘
//! ```
//!
//! Each downstream sink is identified by a unique `SinkId` and has configurable
//! buffering and drop policy.

#![allow(dead_code)]

use std::collections::HashMap;

/// Unique identifier for a relay sink.
pub type SinkId = String;

/// Unique identifier for an upstream source tracked by the relay.
pub type RelaySourceId = String;

/// Policy applied when a downstream sink's buffer is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropPolicy {
    /// Drop the oldest frame in the buffer to make room for the new one.
    DropOldest,
    /// Drop the incoming frame (new frame is discarded).
    DropNewest,
    /// Block until buffer space is available (synchronous only).
    Block,
}

impl Default for DropPolicy {
    fn default() -> Self {
        Self::DropOldest
    }
}

/// A relayed media frame.
#[derive(Debug, Clone)]
pub struct RelayFrame {
    /// The source stream the frame originated from.
    pub source_id: RelaySourceId,
    /// Frame sequence number (monotonically increasing per source).
    pub seq: u64,
    /// Presentation timestamp in microseconds.
    pub pts_us: u64,
    /// Raw payload bytes (caller-defined encoding).
    pub data: Vec<u8>,
}

/// Statistics tracked per downstream sink.
#[derive(Debug, Clone, Default)]
pub struct SinkStats {
    /// Total frames delivered.
    pub frames_delivered: u64,
    /// Total frames dropped due to buffer overflow.
    pub frames_dropped: u64,
    /// Total bytes delivered.
    pub bytes_delivered: u64,
}

/// A downstream sink registered with the relay.
#[derive(Debug)]
pub struct RelaySink {
    /// Sink identifier.
    pub id: SinkId,
    /// Maximum number of frames to buffer.
    pub buffer_capacity: usize,
    /// Overflow policy.
    pub drop_policy: DropPolicy,
    /// Buffered frames waiting to be consumed.
    pub buffer: std::collections::VecDeque<RelayFrame>,
    /// Per-sink statistics.
    pub stats: SinkStats,
    /// Set of source IDs this sink subscribes to (`None` = subscribe to all).
    pub subscribed_sources: Option<Vec<RelaySourceId>>,
    /// Whether the sink is currently active.
    pub active: bool,
}

impl RelaySink {
    /// Creates a new sink with the given buffer capacity and drop policy.
    #[must_use]
    pub fn new(id: SinkId, buffer_capacity: usize, drop_policy: DropPolicy) -> Self {
        Self {
            id,
            buffer_capacity,
            drop_policy,
            buffer: std::collections::VecDeque::new(),
            stats: SinkStats::default(),
            subscribed_sources: None,
            active: true,
        }
    }

    /// Subscribes this sink to a specific set of source IDs.
    pub fn subscribe_to(mut self, sources: Vec<RelaySourceId>) -> Self {
        self.subscribed_sources = Some(sources);
        self
    }

    /// Returns `true` if this sink should receive frames from the given source.
    #[must_use]
    pub fn is_subscribed_to(&self, source_id: &str) -> bool {
        match &self.subscribed_sources {
            None => true,
            Some(subs) => subs.iter().any(|s| s == source_id),
        }
    }

    /// Pushes a frame into the sink buffer, applying the drop policy if full.
    ///
    /// Returns `true` if the frame was buffered, `false` if it was dropped.
    pub fn push(&mut self, frame: RelayFrame) -> bool {
        if self.buffer.len() >= self.buffer_capacity {
            match self.drop_policy {
                DropPolicy::DropOldest => {
                    self.buffer.pop_front();
                    self.stats.frames_dropped += 1;
                }
                DropPolicy::DropNewest => {
                    self.stats.frames_dropped += 1;
                    return false;
                }
                DropPolicy::Block => {
                    // In async context this would yield; in sync tests we block
                    // by discarding the oldest (fallback to DropOldest behaviour).
                    self.buffer.pop_front();
                    self.stats.frames_dropped += 1;
                }
            }
        }
        let byte_count = frame.data.len() as u64;
        self.buffer.push_back(frame);
        self.stats.frames_delivered += 1;
        self.stats.bytes_delivered += byte_count;
        true
    }

    /// Pops the next frame from the buffer, if any.
    pub fn pop(&mut self) -> Option<RelayFrame> {
        self.buffer.pop_front()
    }

    /// Returns the number of frames currently buffered.
    #[must_use]
    pub fn buffered_count(&self) -> usize {
        self.buffer.len()
    }
}

/// Per-source statistics tracked by the relay.
#[derive(Debug, Clone, Default)]
pub struct SourceStats {
    /// Total frames received from this source.
    pub frames_received: u64,
    /// Total frames relayed to at least one sink.
    pub frames_relayed: u64,
    /// Total bytes received.
    pub bytes_received: u64,
}

/// Error type for relay operations.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum RelayError {
    /// Sink with that ID already exists.
    #[error("sink '{0}' already registered")]
    SinkAlreadyExists(String),
    /// Sink not found.
    #[error("sink '{0}' not found")]
    SinkNotFound(String),
    /// Relay has been shut down.
    #[error("relay is shut down")]
    ShutDown,
}

/// Result type for relay operations.
pub type RelayResult<T> = Result<T, RelayError>;

/// A stream relay that fans out frames from upstream sources to downstream sinks.
#[derive(Debug, Default)]
pub struct StreamRelay {
    /// Downstream sinks indexed by ID.
    sinks: HashMap<SinkId, RelaySink>,
    /// Per-source statistics.
    source_stats: HashMap<RelaySourceId, SourceStats>,
    /// Whether the relay is active.
    running: bool,
    /// Total frames relayed across all sinks.
    total_relayed: u64,
}

impl StreamRelay {
    /// Creates a new relay in the running state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sinks: HashMap::new(),
            source_stats: HashMap::new(),
            running: true,
            total_relayed: 0,
        }
    }

    /// Registers a downstream sink.
    pub fn add_sink(&mut self, sink: RelaySink) -> RelayResult<()> {
        if self.sinks.contains_key(&sink.id) {
            return Err(RelayError::SinkAlreadyExists(sink.id.clone()));
        }
        self.sinks.insert(sink.id.clone(), sink);
        Ok(())
    }

    /// Removes a downstream sink by ID.
    pub fn remove_sink(&mut self, sink_id: &str) -> RelayResult<()> {
        self.sinks
            .remove(sink_id)
            .map(|_| ())
            .ok_or_else(|| RelayError::SinkNotFound(sink_id.to_owned()))
    }

    /// Returns a reference to a sink.
    pub fn get_sink(&self, sink_id: &str) -> RelayResult<&RelaySink> {
        self.sinks
            .get(sink_id)
            .ok_or_else(|| RelayError::SinkNotFound(sink_id.to_owned()))
    }

    /// Returns a mutable reference to a sink.
    pub fn get_sink_mut(&mut self, sink_id: &str) -> RelayResult<&mut RelaySink> {
        self.sinks
            .get_mut(sink_id)
            .ok_or_else(|| RelayError::SinkNotFound(sink_id.to_owned()))
    }

    /// Relays an incoming frame to all active, subscribed sinks.
    ///
    /// Returns the number of sinks that received the frame.
    pub fn relay_frame(&mut self, frame: RelayFrame) -> RelayResult<usize> {
        if !self.running {
            return Err(RelayError::ShutDown);
        }

        let stats = self
            .source_stats
            .entry(frame.source_id.clone())
            .or_default();
        stats.frames_received += 1;
        stats.bytes_received += frame.data.len() as u64;

        let mut delivered = 0usize;
        let source_id = frame.source_id.clone();

        for sink in self.sinks.values_mut() {
            if !sink.active || !sink.is_subscribed_to(&source_id) {
                continue;
            }
            if sink.push(frame.clone()) {
                delivered += 1;
            }
        }

        if delivered > 0 {
            let stats = self.source_stats.entry(source_id).or_default();
            stats.frames_relayed += 1;
            self.total_relayed += 1;
        }

        Ok(delivered)
    }

    /// Shuts down the relay (no more frames will be accepted).
    pub fn shutdown(&mut self) {
        self.running = false;
    }

    /// Returns `true` if the relay is running.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Returns total frames relayed (across all sinks).
    #[must_use]
    pub fn total_relayed(&self) -> u64 {
        self.total_relayed
    }

    /// Returns the number of registered sinks.
    #[must_use]
    pub fn sink_count(&self) -> usize {
        self.sinks.len()
    }

    /// Returns per-source statistics.
    #[must_use]
    pub fn source_stats(&self, source_id: &str) -> Option<&SourceStats> {
        self.source_stats.get(source_id)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(source_id: &str, seq: u64, data_len: usize) -> RelayFrame {
        RelayFrame {
            source_id: source_id.to_owned(),
            seq,
            pts_us: seq * 16666,
            data: vec![0u8; data_len],
        }
    }

    #[test]
    fn test_relay_basic() {
        let mut relay = StreamRelay::new();
        let sink = RelaySink::new("sink1".into(), 10, DropPolicy::DropOldest);
        relay.add_sink(sink).unwrap();
        let delivered = relay.relay_frame(make_frame("cam1", 0, 100)).unwrap();
        assert_eq!(delivered, 1);
        assert_eq!(relay.total_relayed(), 1);
    }

    #[test]
    fn test_relay_to_multiple_sinks() {
        let mut relay = StreamRelay::new();
        for i in 0..3 {
            let sink = RelaySink::new(format!("sink{i}"), 10, DropPolicy::DropOldest);
            relay.add_sink(sink).unwrap();
        }
        let delivered = relay.relay_frame(make_frame("cam1", 0, 50)).unwrap();
        assert_eq!(delivered, 3);
    }

    #[test]
    fn test_source_subscription_filter() {
        let mut relay = StreamRelay::new();
        let sink = RelaySink::new("sink1".into(), 10, DropPolicy::DropOldest)
            .subscribe_to(vec!["cam1".to_owned()]);
        relay.add_sink(sink).unwrap();

        // cam1 → subscribed
        let d1 = relay.relay_frame(make_frame("cam1", 0, 10)).unwrap();
        assert_eq!(d1, 1);

        // cam2 → not subscribed
        let d2 = relay.relay_frame(make_frame("cam2", 0, 10)).unwrap();
        assert_eq!(d2, 0);
    }

    #[test]
    fn test_drop_oldest_policy() {
        let sink = RelaySink::new("s".into(), 2, DropPolicy::DropOldest);
        let mut relay = StreamRelay::new();
        relay.add_sink(sink).unwrap();

        relay.relay_frame(make_frame("c", 0, 10)).unwrap();
        relay.relay_frame(make_frame("c", 1, 10)).unwrap();
        relay.relay_frame(make_frame("c", 2, 10)).unwrap(); // should drop seq=0

        let sink = relay.get_sink("s").unwrap();
        assert_eq!(sink.buffered_count(), 2);
        assert_eq!(sink.stats.frames_dropped, 1);
    }

    #[test]
    fn test_drop_newest_policy() {
        let sink = RelaySink::new("s".into(), 2, DropPolicy::DropNewest);
        let mut relay = StreamRelay::new();
        relay.add_sink(sink).unwrap();

        relay.relay_frame(make_frame("c", 0, 10)).unwrap();
        relay.relay_frame(make_frame("c", 1, 10)).unwrap();
        let delivered = relay.relay_frame(make_frame("c", 2, 10)).unwrap();
        assert_eq!(delivered, 0); // new frame dropped

        let sink = relay.get_sink("s").unwrap();
        assert_eq!(sink.buffered_count(), 2);
        assert_eq!(sink.stats.frames_dropped, 1);
    }

    #[test]
    fn test_pop_from_sink() {
        let mut relay = StreamRelay::new();
        relay.add_sink(RelaySink::new("s".into(), 5, DropPolicy::DropOldest)).unwrap();
        relay.relay_frame(make_frame("c", 42, 20)).unwrap();
        let frame = relay.get_sink_mut("s").unwrap().pop().unwrap();
        assert_eq!(frame.seq, 42);
    }

    #[test]
    fn test_remove_sink() {
        let mut relay = StreamRelay::new();
        relay.add_sink(RelaySink::new("s".into(), 5, DropPolicy::DropOldest)).unwrap();
        relay.remove_sink("s").unwrap();
        assert_eq!(relay.sink_count(), 0);
    }

    #[test]
    fn test_shutdown_prevents_relay() {
        let mut relay = StreamRelay::new();
        relay.shutdown();
        let result = relay.relay_frame(make_frame("c", 0, 10));
        assert!(matches!(result, Err(RelayError::ShutDown)));
    }

    #[test]
    fn test_duplicate_sink_rejected() {
        let mut relay = StreamRelay::new();
        relay.add_sink(RelaySink::new("s".into(), 5, DropPolicy::DropOldest)).unwrap();
        let result = relay.add_sink(RelaySink::new("s".into(), 5, DropPolicy::DropOldest));
        assert!(matches!(result, Err(RelayError::SinkAlreadyExists(_))));
    }

    #[test]
    fn test_source_stats_updated() {
        let mut relay = StreamRelay::new();
        relay.add_sink(RelaySink::new("s".into(), 5, DropPolicy::DropOldest)).unwrap();
        relay.relay_frame(make_frame("cam", 0, 500)).unwrap();
        let stats = relay.source_stats("cam").unwrap();
        assert_eq!(stats.frames_received, 1);
        assert_eq!(stats.bytes_received, 500);
    }
}
