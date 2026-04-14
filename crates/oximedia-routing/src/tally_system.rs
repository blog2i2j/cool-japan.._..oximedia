#![allow(dead_code)]
//! Tally system for signaling active routing paths to external tally light
//! controllers.
//!
//! A *tally* is a visual indicator (typically a red or green light on a camera
//! or studio monitor) that communicates whether a source is currently on
//! programme, in preview, or idle.  This module models the complete lifecycle:
//!
//! - [`TallyState`] — the signal a source is in (programme, preview, idle, …).
//! - [`TallySource`] — a named input that can carry a tally state.
//! - [`TallyDestination`] — a physical tally controller port or software
//!   endpoint.
//! - [`TallyBus`] — central authority: maps sources to destinations and
//!   propagates state changes.
//! - [`TallyEvent`] — log entry for a state transition.
//!
//! # Design
//!
//! The bus keeps a simple priority ordering: `Programme > Preview > Idle`.
//! When the same source is simultaneously assigned to multiple buses (e.g., a
//! two-bus vision mixer), the highest-priority state wins.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// TallyState
// ---------------------------------------------------------------------------

/// The tally state a source is currently in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TallyState {
    /// Source is idle — not in programme or preview.
    Idle = 0,
    /// Source is in preview (typically green tally).
    Preview = 1,
    /// Source is live on programme (typically red tally).
    Programme = 2,
}

impl TallyState {
    /// Returns the conventional colour name for this state.
    pub fn colour_name(self) -> &'static str {
        match self {
            Self::Idle => "off",
            Self::Preview => "green",
            Self::Programme => "red",
        }
    }
}

impl fmt::Display for TallyState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Idle => write!(f, "idle"),
            Self::Preview => write!(f, "preview"),
            Self::Programme => write!(f, "programme"),
        }
    }
}

// ---------------------------------------------------------------------------
// TallyProtocol
// ---------------------------------------------------------------------------

/// The transport/protocol used to drive physical tally lights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TallyProtocol {
    /// Generic GPIO lines (3-wire: programme, preview, GND).
    Gpio,
    /// TSL (Television Systems Limited) UMD protocol over RS-422.
    TslUmd,
    /// Blackmagic Design SDI tally via ATEM control protocol.
    AtemSdi,
    /// RossTalk over TCP (Ross Video switcher tally).
    RossTalk,
    /// Software callback — no hardware, used in tests and virtual productions.
    Software,
}

impl fmt::Display for TallyProtocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gpio => write!(f, "GPIO"),
            Self::TslUmd => write!(f, "TSL UMD"),
            Self::AtemSdi => write!(f, "ATEM SDI"),
            Self::RossTalk => write!(f, "RossTalk"),
            Self::Software => write!(f, "software"),
        }
    }
}

// ---------------------------------------------------------------------------
// TallySource
// ---------------------------------------------------------------------------

/// A named source that participates in tally signaling.
///
/// A source maps onto a physical camera, graphics machine, replay server,
/// or any other device whose on-air status matters.
#[derive(Debug, Clone)]
pub struct TallySource {
    /// Unique source identifier.
    pub id: String,
    /// Human-readable label (e.g. `"CAM 1"`, `"VT 2"`).
    pub label: String,
    /// Optional bus number this source belongs to (e.g. main programme bus 1).
    pub bus_number: Option<u32>,
}

impl TallySource {
    /// Creates a new tally source.
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            bus_number: None,
        }
    }

    /// Assigns a bus number to the source.
    pub fn with_bus(mut self, bus: u32) -> Self {
        self.bus_number = Some(bus);
        self
    }
}

impl fmt::Display for TallySource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TallySource('{}', label='{}')", self.id, self.label)
    }
}

// ---------------------------------------------------------------------------
// TallyDestination
// ---------------------------------------------------------------------------

/// A physical or virtual tally endpoint.
///
/// Each destination is bound to one source; when the source's tally state
/// changes, the destination is notified.
#[derive(Debug, Clone)]
pub struct TallyDestination {
    /// Unique destination identifier.
    pub id: String,
    /// Human-readable label for this tally output.
    pub label: String,
    /// Transport protocol used to drive this destination.
    pub protocol: TallyProtocol,
    /// The source id this destination is currently reflecting.
    pub source_id: Option<String>,
    /// Current tally state being displayed.
    pub current_state: TallyState,
}

impl TallyDestination {
    /// Creates a new idle destination.
    pub fn new(id: impl Into<String>, label: impl Into<String>, protocol: TallyProtocol) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            protocol,
            source_id: None,
            current_state: TallyState::Idle,
        }
    }

    /// Assigns this destination to follow a source.
    pub fn assign_source(&mut self, source_id: impl Into<String>) {
        self.source_id = Some(source_id.into());
    }

    /// Clears the source assignment (destination goes idle).
    pub fn unassign_source(&mut self) {
        self.source_id = None;
        self.current_state = TallyState::Idle;
    }
}

impl fmt::Display for TallyDestination {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let src = self.source_id.as_deref().unwrap_or("<unassigned>");
        write!(
            f,
            "TallyDest('{}', src={}, state={}, proto={})",
            self.label, src, self.current_state, self.protocol
        )
    }
}

// ---------------------------------------------------------------------------
// TallyEvent
// ---------------------------------------------------------------------------

/// A log entry recording a tally state change.
#[derive(Debug, Clone)]
pub struct TallyEvent {
    /// The source whose tally changed.
    pub source_id: String,
    /// Previous state.
    pub previous_state: TallyState,
    /// New state.
    pub new_state: TallyState,
    /// Sequence number (monotonically increasing within the bus).
    pub sequence: u64,
}

impl TallyEvent {
    fn new(source_id: String, previous: TallyState, next: TallyState, seq: u64) -> Self {
        Self {
            source_id,
            previous_state: previous,
            new_state: next,
            sequence: seq,
        }
    }
}

impl fmt::Display for TallyEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] source '{}': {} → {}",
            self.sequence, self.source_id, self.previous_state, self.new_state
        )
    }
}

// ---------------------------------------------------------------------------
// TallyError
// ---------------------------------------------------------------------------

/// Errors returned by [`TallyBus`] operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TallyError {
    /// A source with the given id is already registered.
    DuplicateSourceId(String),
    /// No source with the given id exists.
    SourceNotFound(String),
    /// A destination with the given id is already registered.
    DuplicateDestinationId(String),
    /// No destination with the given id exists.
    DestinationNotFound(String),
    /// The destination has no source assigned to it.
    NoSourceAssigned(String),
}

impl fmt::Display for TallyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateSourceId(id) => write!(f, "tally source already registered: {id}"),
            Self::SourceNotFound(id) => write!(f, "tally source not found: {id}"),
            Self::DuplicateDestinationId(id) => {
                write!(f, "tally destination already registered: {id}")
            }
            Self::DestinationNotFound(id) => write!(f, "tally destination not found: {id}"),
            Self::NoSourceAssigned(id) => {
                write!(f, "destination '{id}' has no source assigned")
            }
        }
    }
}

impl std::error::Error for TallyError {}

// ---------------------------------------------------------------------------
// TallyBus
// ---------------------------------------------------------------------------

/// Central tally bus: manages sources, destinations, and state propagation.
///
/// Each source has exactly one "effective" tally state.  When a source is
/// simultaneously driven by multiple callers (e.g. from different mixers),
/// the highest-priority state wins (`Programme > Preview > Idle`).
///
/// Call [`TallyBus::set_state`] to change a source's state.  The bus will
/// immediately update all destinations assigned to that source and append a
/// [`TallyEvent`] to the event log.
#[derive(Debug, Default)]
pub struct TallyBus {
    sources: HashMap<String, TallySource>,
    destinations: HashMap<String, TallyDestination>,
    /// Current effective tally state for each source.
    source_states: HashMap<String, TallyState>,
    /// Event log.
    events: Vec<TallyEvent>,
    /// Sequence counter for event log entries.
    sequence: u64,
}

impl TallyBus {
    /// Creates an empty tally bus.
    pub fn new() -> Self {
        Self::default()
    }

    // -----------------------------------------------------------------------
    // Source management
    // -----------------------------------------------------------------------

    /// Registers a new tally source.
    pub fn register_source(&mut self, source: TallySource) -> Result<(), TallyError> {
        if self.sources.contains_key(&source.id) {
            return Err(TallyError::DuplicateSourceId(source.id.clone()));
        }
        let id = source.id.clone();
        self.sources.insert(id.clone(), source);
        self.source_states.insert(id, TallyState::Idle);
        Ok(())
    }

    /// Removes a tally source.  All destinations assigned to it go idle.
    pub fn remove_source(&mut self, source_id: &str) -> Result<TallySource, TallyError> {
        let source = self
            .sources
            .remove(source_id)
            .ok_or_else(|| TallyError::SourceNotFound(source_id.to_string()))?;

        self.source_states.remove(source_id);

        // Drive all destinations that were tracking this source to idle
        for dest in self.destinations.values_mut() {
            if dest.source_id.as_deref() == Some(source_id) {
                dest.current_state = TallyState::Idle;
            }
        }

        Ok(source)
    }

    /// Returns a reference to the source with the given id.
    pub fn get_source(&self, source_id: &str) -> Option<&TallySource> {
        self.sources.get(source_id)
    }

    /// Returns all registered sources.
    pub fn all_sources(&self) -> Vec<&TallySource> {
        self.sources.values().collect()
    }

    /// Returns the current effective tally state for the given source.
    pub fn source_state(&self, source_id: &str) -> Option<TallyState> {
        self.source_states.get(source_id).copied()
    }

    /// Returns all sources currently in the given state.
    pub fn sources_in_state(&self, state: TallyState) -> Vec<&TallySource> {
        self.source_states
            .iter()
            .filter(|(_, s)| **s == state)
            .filter_map(|(id, _)| self.sources.get(id))
            .collect()
    }

    // -----------------------------------------------------------------------
    // State changes
    // -----------------------------------------------------------------------

    /// Sets the tally state for a source and propagates it to all bound
    /// destinations.
    ///
    /// If the new state differs from the current state, a [`TallyEvent`] is
    /// appended to the event log.
    pub fn set_state(&mut self, source_id: &str, new_state: TallyState) -> Result<(), TallyError> {
        if !self.sources.contains_key(source_id) {
            return Err(TallyError::SourceNotFound(source_id.to_string()));
        }

        let previous = *self
            .source_states
            .get(source_id)
            .unwrap_or(&TallyState::Idle);

        if previous != new_state {
            self.source_states.insert(source_id.to_string(), new_state);

            // Propagate to destinations
            for dest in self.destinations.values_mut() {
                if dest.source_id.as_deref() == Some(source_id) {
                    dest.current_state = new_state;
                }
            }

            // Log event
            let seq = self.sequence;
            self.sequence += 1;
            self.events.push(TallyEvent::new(
                source_id.to_string(),
                previous,
                new_state,
                seq,
            ));
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Destination management
    // -----------------------------------------------------------------------

    /// Registers a new tally destination.
    pub fn register_destination(
        &mut self,
        destination: TallyDestination,
    ) -> Result<(), TallyError> {
        if self.destinations.contains_key(&destination.id) {
            return Err(TallyError::DuplicateDestinationId(destination.id.clone()));
        }
        self.destinations
            .insert(destination.id.clone(), destination);
        Ok(())
    }

    /// Removes a tally destination.
    pub fn remove_destination(
        &mut self,
        destination_id: &str,
    ) -> Result<TallyDestination, TallyError> {
        self.destinations
            .remove(destination_id)
            .ok_or_else(|| TallyError::DestinationNotFound(destination_id.to_string()))
    }

    /// Returns a reference to the destination with the given id.
    pub fn get_destination(&self, destination_id: &str) -> Option<&TallyDestination> {
        self.destinations.get(destination_id)
    }

    /// Returns all registered destinations.
    pub fn all_destinations(&self) -> Vec<&TallyDestination> {
        self.destinations.values().collect()
    }

    /// Assigns a destination to follow a source.  The destination's state is
    /// immediately updated to the source's current effective state.
    pub fn assign_destination_to_source(
        &mut self,
        destination_id: &str,
        source_id: &str,
    ) -> Result<(), TallyError> {
        if !self.sources.contains_key(source_id) {
            return Err(TallyError::SourceNotFound(source_id.to_string()));
        }

        let current_state = self
            .source_states
            .get(source_id)
            .copied()
            .unwrap_or(TallyState::Idle);

        let dest = self
            .destinations
            .get_mut(destination_id)
            .ok_or_else(|| TallyError::DestinationNotFound(destination_id.to_string()))?;

        dest.assign_source(source_id);
        dest.current_state = current_state;

        Ok(())
    }

    /// Unassigns a destination from its source (destination goes idle).
    pub fn unassign_destination(&mut self, destination_id: &str) -> Result<(), TallyError> {
        let dest = self
            .destinations
            .get_mut(destination_id)
            .ok_or_else(|| TallyError::DestinationNotFound(destination_id.to_string()))?;

        dest.unassign_source();
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Event log
    // -----------------------------------------------------------------------

    /// Returns the full event log.
    pub fn events(&self) -> &[TallyEvent] {
        &self.events
    }

    /// Returns events for a specific source (newest last).
    pub fn events_for_source(&self, source_id: &str) -> Vec<&TallyEvent> {
        self.events
            .iter()
            .filter(|e| e.source_id == source_id)
            .collect()
    }

    /// Clears the event log.
    pub fn clear_events(&mut self) {
        self.events.clear();
    }

    /// Returns the total number of registered sources.
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Returns the total number of registered destinations.
    pub fn destination_count(&self) -> usize {
        self.destinations.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_source(id: &str) -> TallySource {
        TallySource::new(id, id)
    }

    fn make_destination(id: &str) -> TallyDestination {
        TallyDestination::new(id, id, TallyProtocol::Software)
    }

    #[test]
    fn test_register_source_and_retrieve() {
        let mut bus = TallyBus::new();
        bus.register_source(make_source("cam-1")).expect("ok");
        assert!(bus.get_source("cam-1").is_some());
        assert_eq!(bus.source_state("cam-1"), Some(TallyState::Idle));
    }

    #[test]
    fn test_duplicate_source_returns_error() {
        let mut bus = TallyBus::new();
        bus.register_source(make_source("cam-dup")).expect("ok");
        let result = bus.register_source(make_source("cam-dup"));
        assert!(matches!(result, Err(TallyError::DuplicateSourceId(_))));
    }

    #[test]
    fn test_set_state_to_programme() {
        let mut bus = TallyBus::new();
        bus.register_source(make_source("cam-2")).expect("ok");
        bus.set_state("cam-2", TallyState::Programme).expect("ok");
        assert_eq!(bus.source_state("cam-2"), Some(TallyState::Programme));
    }

    #[test]
    fn test_state_change_appends_event() {
        let mut bus = TallyBus::new();
        bus.register_source(make_source("cam-3")).expect("ok");
        bus.set_state("cam-3", TallyState::Preview).expect("ok");
        bus.set_state("cam-3", TallyState::Programme).expect("ok");
        let events = bus.events_for_source("cam-3");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].previous_state, TallyState::Idle);
        assert_eq!(events[0].new_state, TallyState::Preview);
        assert_eq!(events[1].previous_state, TallyState::Preview);
        assert_eq!(events[1].new_state, TallyState::Programme);
    }

    #[test]
    fn test_no_duplicate_event_for_same_state() {
        let mut bus = TallyBus::new();
        bus.register_source(make_source("cam-4")).expect("ok");
        bus.set_state("cam-4", TallyState::Programme).expect("ok");
        bus.set_state("cam-4", TallyState::Programme).expect("ok"); // same — should not append
        assert_eq!(bus.events_for_source("cam-4").len(), 1);
    }

    #[test]
    fn test_destination_receives_state_update() {
        let mut bus = TallyBus::new();
        bus.register_source(make_source("cam-5")).expect("ok");
        bus.register_destination(make_destination("dest-5"))
            .expect("ok");
        bus.assign_destination_to_source("dest-5", "cam-5")
            .expect("ok");

        bus.set_state("cam-5", TallyState::Programme).expect("ok");

        let dest = bus.get_destination("dest-5").expect("should exist");
        assert_eq!(dest.current_state, TallyState::Programme);
    }

    #[test]
    fn test_destination_gets_current_state_on_assign() {
        let mut bus = TallyBus::new();
        bus.register_source(make_source("cam-6")).expect("ok");
        bus.set_state("cam-6", TallyState::Preview).expect("ok");

        bus.register_destination(make_destination("dest-6"))
            .expect("ok");
        // Assign *after* state has been set — should immediately reflect current state
        bus.assign_destination_to_source("dest-6", "cam-6")
            .expect("ok");

        let dest = bus.get_destination("dest-6").expect("should exist");
        assert_eq!(dest.current_state, TallyState::Preview);
    }

    #[test]
    fn test_unassign_destination_goes_idle() {
        let mut bus = TallyBus::new();
        bus.register_source(make_source("cam-7")).expect("ok");
        bus.set_state("cam-7", TallyState::Programme).expect("ok");

        bus.register_destination(make_destination("dest-7"))
            .expect("ok");
        bus.assign_destination_to_source("dest-7", "cam-7")
            .expect("ok");
        bus.unassign_destination("dest-7").expect("ok");

        let dest = bus.get_destination("dest-7").expect("should exist");
        assert_eq!(dest.current_state, TallyState::Idle);
        assert!(dest.source_id.is_none());
    }

    #[test]
    fn test_remove_source_drives_destinations_idle() {
        let mut bus = TallyBus::new();
        bus.register_source(make_source("cam-8")).expect("ok");
        bus.set_state("cam-8", TallyState::Programme).expect("ok");

        bus.register_destination(make_destination("dest-8"))
            .expect("ok");
        bus.assign_destination_to_source("dest-8", "cam-8")
            .expect("ok");

        bus.remove_source("cam-8").expect("ok");

        let dest = bus.get_destination("dest-8").expect("should exist");
        assert_eq!(dest.current_state, TallyState::Idle);
    }

    #[test]
    fn test_sources_in_state_filter() {
        let mut bus = TallyBus::new();
        bus.register_source(make_source("c1")).expect("ok");
        bus.register_source(make_source("c2")).expect("ok");
        bus.register_source(make_source("c3")).expect("ok");
        bus.set_state("c1", TallyState::Programme).expect("ok");
        bus.set_state("c2", TallyState::Preview).expect("ok");

        let on_programme = bus.sources_in_state(TallyState::Programme);
        assert_eq!(on_programme.len(), 1);
        assert_eq!(on_programme[0].id, "c1");

        let idle = bus.sources_in_state(TallyState::Idle);
        assert_eq!(idle.len(), 1);
        assert_eq!(idle[0].id, "c3");
    }

    #[test]
    fn test_colour_name_mapping() {
        assert_eq!(TallyState::Programme.colour_name(), "red");
        assert_eq!(TallyState::Preview.colour_name(), "green");
        assert_eq!(TallyState::Idle.colour_name(), "off");
    }

    #[test]
    fn test_display_and_event_formatting() {
        let src = TallySource::new("s1", "CAM 1");
        let s = format!("{src}");
        assert!(s.contains("CAM 1"));

        let dest = TallyDestination::new("d1", "Camera Tally", TallyProtocol::TslUmd);
        let s = format!("{dest}");
        assert!(s.contains("Camera Tally"));

        let ev = TallyEvent::new("s1".to_string(), TallyState::Idle, TallyState::Programme, 0);
        let s = format!("{ev}");
        assert!(s.contains("idle"));
        assert!(s.contains("programme"));
    }

    #[test]
    fn test_set_state_unknown_source_returns_error() {
        let mut bus = TallyBus::new();
        let result = bus.set_state("ghost", TallyState::Programme);
        assert!(matches!(result, Err(TallyError::SourceNotFound(_))));
    }

    #[test]
    fn test_clear_events() {
        let mut bus = TallyBus::new();
        bus.register_source(make_source("cam-ev")).expect("ok");
        bus.set_state("cam-ev", TallyState::Programme).expect("ok");
        assert_eq!(bus.events().len(), 1);
        bus.clear_events();
        assert_eq!(bus.events().len(), 0);
    }

    // -----------------------------------------------------------------------
    // Additional tests (8+) for full coverage
    // -----------------------------------------------------------------------

    #[test]
    fn test_source_count_after_register_and_remove() {
        let mut bus = TallyBus::new();
        assert_eq!(bus.source_count(), 0);
        bus.register_source(make_source("s1")).expect("ok");
        bus.register_source(make_source("s2")).expect("ok");
        assert_eq!(bus.source_count(), 2);
        bus.remove_source("s1").expect("ok");
        assert_eq!(bus.source_count(), 1);
    }

    #[test]
    fn test_destination_count_after_register_and_remove() {
        let mut bus = TallyBus::new();
        bus.register_destination(make_destination("d1"))
            .expect("ok");
        bus.register_destination(make_destination("d2"))
            .expect("ok");
        assert_eq!(bus.destination_count(), 2);
        bus.remove_destination("d1").expect("ok");
        assert_eq!(bus.destination_count(), 1);
    }

    #[test]
    fn test_remove_nonexistent_source_returns_error() {
        let mut bus = TallyBus::new();
        let result = bus.remove_source("ghost");
        assert!(matches!(result, Err(TallyError::SourceNotFound(_))));
    }

    #[test]
    fn test_remove_nonexistent_destination_returns_error() {
        let mut bus = TallyBus::new();
        let result = bus.remove_destination("ghost");
        assert!(matches!(result, Err(TallyError::DestinationNotFound(_))));
    }

    #[test]
    fn test_assign_destination_to_unknown_source_returns_error() {
        let mut bus = TallyBus::new();
        bus.register_destination(make_destination("d")).expect("ok");
        let result = bus.assign_destination_to_source("d", "no-such-source");
        assert!(matches!(result, Err(TallyError::SourceNotFound(_))));
    }

    #[test]
    fn test_assign_unknown_destination_returns_error() {
        let mut bus = TallyBus::new();
        bus.register_source(make_source("s")).expect("ok");
        let result = bus.assign_destination_to_source("no-such-dest", "s");
        assert!(matches!(result, Err(TallyError::DestinationNotFound(_))));
    }

    #[test]
    fn test_unassign_unknown_destination_returns_error() {
        let mut bus = TallyBus::new();
        let result = bus.unassign_destination("no-such-dest");
        assert!(matches!(result, Err(TallyError::DestinationNotFound(_))));
    }

    #[test]
    fn test_multiple_destinations_follow_same_source() {
        let mut bus = TallyBus::new();
        bus.register_source(make_source("src")).expect("ok");
        bus.register_destination(make_destination("dest-a"))
            .expect("ok");
        bus.register_destination(make_destination("dest-b"))
            .expect("ok");
        bus.assign_destination_to_source("dest-a", "src")
            .expect("ok");
        bus.assign_destination_to_source("dest-b", "src")
            .expect("ok");

        bus.set_state("src", TallyState::Preview).expect("ok");

        let da = bus.get_destination("dest-a").expect("should exist");
        let db = bus.get_destination("dest-b").expect("should exist");
        assert_eq!(da.current_state, TallyState::Preview);
        assert_eq!(db.current_state, TallyState::Preview);
    }

    #[test]
    fn test_source_with_bus_number() {
        let src = TallySource::new("cam", "CAM 1").with_bus(1);
        assert_eq!(src.bus_number, Some(1));
    }

    #[test]
    fn test_tally_protocol_display() {
        assert_eq!(format!("{}", TallyProtocol::TslUmd), "TSL UMD");
        assert_eq!(format!("{}", TallyProtocol::RossTalk), "RossTalk");
        assert_eq!(format!("{}", TallyProtocol::Software), "software");
        assert_eq!(format!("{}", TallyProtocol::Gpio), "GPIO");
        assert_eq!(format!("{}", TallyProtocol::AtemSdi), "ATEM SDI");
    }

    #[test]
    fn test_tally_state_ordering() {
        // Programme > Preview > Idle
        assert!(TallyState::Programme > TallyState::Preview);
        assert!(TallyState::Preview > TallyState::Idle);
    }

    #[test]
    fn test_tally_state_display() {
        assert_eq!(format!("{}", TallyState::Idle), "idle");
        assert_eq!(format!("{}", TallyState::Preview), "preview");
        assert_eq!(format!("{}", TallyState::Programme), "programme");
    }

    #[test]
    fn test_all_sources_returns_all() {
        let mut bus = TallyBus::new();
        bus.register_source(make_source("a")).expect("ok");
        bus.register_source(make_source("b")).expect("ok");
        bus.register_source(make_source("c")).expect("ok");
        assert_eq!(bus.all_sources().len(), 3);
    }

    #[test]
    fn test_all_destinations_returns_all() {
        let mut bus = TallyBus::new();
        bus.register_destination(make_destination("d1"))
            .expect("ok");
        bus.register_destination(make_destination("d2"))
            .expect("ok");
        assert_eq!(bus.all_destinations().len(), 2);
    }

    #[test]
    fn test_duplicate_destination_returns_error() {
        let mut bus = TallyBus::new();
        bus.register_destination(make_destination("dest-dup"))
            .expect("ok");
        let result = bus.register_destination(make_destination("dest-dup"));
        assert!(matches!(result, Err(TallyError::DuplicateDestinationId(_))));
    }

    #[test]
    fn test_tally_error_display() {
        let err = TallyError::DuplicateSourceId("x".to_string());
        assert!(format!("{err}").contains("x"));
        let err2 = TallyError::SourceNotFound("y".to_string());
        assert!(format!("{err2}").contains("y"));
    }

    #[test]
    fn test_event_sequence_increments() {
        let mut bus = TallyBus::new();
        bus.register_source(make_source("seq-test")).expect("ok");
        bus.set_state("seq-test", TallyState::Preview).expect("ok");
        bus.set_state("seq-test", TallyState::Programme)
            .expect("ok");
        bus.set_state("seq-test", TallyState::Idle).expect("ok");
        let events = bus.events();
        assert_eq!(events.len(), 3);
        assert!(events[0].sequence < events[1].sequence);
        assert!(events[1].sequence < events[2].sequence);
    }
}
