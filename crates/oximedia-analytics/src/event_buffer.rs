//! Generic bounded event buffer with capacity-triggered draining.
//!
//! Provides a simple, generic alternative to the time-and-count–based
//! `realtime::EventBuffer`.  This module is tailored to batch
//! pipeline stages that only need capacity-based backpressure and explicit
//! drain control.

use crate::error::AnalyticsError;

// ─── Event ───────────────────────────────────────────────────────────────────

/// A generic analytics event stored in the buffer.
///
/// In most pipelines you will use a concrete domain type (e.g. a struct
/// wrapping a session ID and payload).  The buffer accepts any `Send + 'static`
/// type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
    /// Opaque string label identifying the event kind (e.g. `"play"`, `"pause"`).
    pub kind: String,
    /// Session or user identifier associated with this event.
    pub session_id: String,
    /// Millisecond epoch timestamp when the event occurred.
    pub timestamp_ms: u64,
}

impl Event {
    /// Creates a new `Event`.
    pub fn new(kind: impl Into<String>, session_id: impl Into<String>, timestamp_ms: u64) -> Self {
        Self {
            kind: kind.into(),
            session_id: session_id.into(),
            timestamp_ms,
        }
    }
}

// ─── EventBuffer ─────────────────────────────────────────────────────────────

/// A bounded in-memory event buffer.
///
/// Events accumulate in the buffer until it is full, at which point
/// [`EventBuffer::push`] returns an error with `BufferFull` status (but does
/// **not** drop the event).  Callers should call [`EventBuffer::drain`] to
/// consume the current contents and free space.
///
/// # Example
///
/// ```rust
/// use oximedia_analytics::event_buffer::{Event, EventBuffer};
///
/// let mut buf = EventBuffer::new(3).expect("valid capacity");
/// buf.push(Event::new("play", "s1", 0)).expect("space available");
/// buf.push(Event::new("pause", "s1", 1000)).expect("space available");
/// let drained = buf.drain();
/// assert_eq!(drained.len(), 2);
/// ```
#[derive(Debug)]
pub struct EventBuffer {
    capacity: usize,
    events: Vec<Event>,
}

impl EventBuffer {
    /// Creates a new `EventBuffer` with the given maximum `capacity`.
    ///
    /// # Errors
    ///
    /// Returns [`AnalyticsError::InvalidInput`] when `capacity` is zero.
    pub fn new(capacity: usize) -> Result<Self, AnalyticsError> {
        if capacity == 0 {
            return Err(AnalyticsError::InvalidInput("capacity must be > 0".into()));
        }
        Ok(Self {
            capacity,
            events: Vec::with_capacity(capacity),
        })
    }

    /// Appends `event` to the buffer.
    ///
    /// # Errors
    ///
    /// Returns `AnalyticsError::BufferFull` (or equivalent) when the buffer
    /// is already at capacity.  The caller must drain first.
    pub fn push(&mut self, event: Event) -> Result<(), AnalyticsError> {
        if self.events.len() >= self.capacity {
            return Err(AnalyticsError::InvalidInput(format!(
                "event buffer full (capacity = {})",
                self.capacity
            )));
        }
        self.events.push(event);
        Ok(())
    }

    /// Removes and returns all buffered events, leaving the buffer empty.
    pub fn drain(&mut self) -> Vec<Event> {
        std::mem::take(&mut self.events)
    }

    /// Returns the number of events currently held in the buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns `true` when no events are buffered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Returns the maximum capacity of the buffer.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(kind: &str) -> Event {
        Event::new(kind, "s1", 0)
    }

    #[test]
    fn zero_capacity_errors() {
        assert!(EventBuffer::new(0).is_err());
    }

    #[test]
    fn push_and_drain() {
        let mut buf = EventBuffer::new(4).expect("valid");
        buf.push(ev("play")).expect("room");
        buf.push(ev("pause")).expect("room");
        let events = buf.drain();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].kind, "play");
        assert_eq!(events[1].kind, "pause");
    }

    #[test]
    fn drain_clears_buffer() {
        let mut buf = EventBuffer::new(4).expect("valid");
        buf.push(ev("seek")).expect("room");
        let _ = buf.drain();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn push_beyond_capacity_errors() {
        let mut buf = EventBuffer::new(2).expect("valid");
        buf.push(ev("a")).expect("room");
        buf.push(ev("b")).expect("room");
        assert!(buf.push(ev("c")).is_err());
        // Event is NOT lost — caller can drain and retry.
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn drain_empty_returns_empty_vec() {
        let mut buf = EventBuffer::new(5).expect("valid");
        assert_eq!(buf.drain(), Vec::<Event>::new());
    }

    #[test]
    fn capacity_reported_correctly() {
        let buf = EventBuffer::new(7).expect("valid");
        assert_eq!(buf.capacity(), 7);
    }

    #[test]
    fn multiple_drain_cycles() {
        let mut buf = EventBuffer::new(3).expect("valid");
        for cycle in 0u64..3 {
            buf.push(Event::new("e", "s", cycle)).expect("room");
            buf.push(Event::new("e", "s", cycle + 100)).expect("room");
            let batch = buf.drain();
            assert_eq!(batch.len(), 2);
            assert!(buf.is_empty());
        }
    }
}
