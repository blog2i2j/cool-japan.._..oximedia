#![allow(dead_code)]
//! Intercom module for point-to-point and party-line communication routing.
//!
//! Broadcast and live-production facilities use intercom (interruptible
//! fold-back / IFB) systems to route voice communications between crew members
//! and on-air talent.  This module models:
//!
//! - [`IntercomPort`] — a physical or virtual intercom station (belt-pack,
//!   headset jack, software client).
//! - [`PartyLine`] — a named group channel; any port assigned to the line hears
//!   all other ports on the same line simultaneously.
//! - [`PointToPointCall`] — a direct, private call between exactly two ports.
//! - [`IntercomMatrix`] — the central routing authority that manages ports,
//!   party lines, and point-to-point calls, and answers "which ports can hear
//!   which other ports?"
//!
//! # Design
//!
//! Audio routing is modelled as *membership sets* rather than signal graphs.
//! A port is a member of zero or more party lines.  When audio is sent from
//! port A, every other port that shares at least one party line with A, plus
//! any port in a direct call with A, receives the audio.  Priorities can be
//! set per port, enabling "forced listen" (executive override) scenarios.

use std::collections::{HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// IntercomPortKind
// ---------------------------------------------------------------------------

/// The physical or logical type of an intercom port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntercomPortKind {
    /// Hardwired beltpack (4-wire or 2-wire).
    Beltpack,
    /// Fixed panel station (e.g. engineering position, director's desk).
    PanelStation,
    /// IFB earpiece feed to on-air talent.
    IfbEarpiece,
    /// Software-based intercom client.
    SoftClient,
    /// Programme audio feed injected into an IFB mix.
    ProgrammeFeed,
    /// Telephone hybrid / POTS interface.
    TelephoneHybrid,
}

impl fmt::Display for IntercomPortKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Beltpack => write!(f, "beltpack"),
            Self::PanelStation => write!(f, "panel station"),
            Self::IfbEarpiece => write!(f, "IFB earpiece"),
            Self::SoftClient => write!(f, "soft client"),
            Self::ProgrammeFeed => write!(f, "programme feed"),
            Self::TelephoneHybrid => write!(f, "telephone hybrid"),
        }
    }
}

// ---------------------------------------------------------------------------
// IntercomPort
// ---------------------------------------------------------------------------

/// A single intercom endpoint.
#[derive(Debug, Clone)]
pub struct IntercomPort {
    /// Unique port identifier.
    pub id: String,
    /// Human-readable label shown on panels and software clients.
    pub label: String,
    /// Physical/logical type.
    pub kind: IntercomPortKind,
    /// Whether the port is currently active (powered on, logged in).
    pub active: bool,
    /// Whether the port's microphone is currently open (key is pressed).
    pub mic_open: bool,
    /// Listen gain in dB (−∞ to +12 dB; 0 = unity).
    pub listen_gain_db: f32,
    /// Whether this port can override all other ports (executive override).
    pub executive_override: bool,
    /// Whether this port is currently muted (by local or remote request).
    pub muted: bool,
}

impl IntercomPort {
    /// Creates a new, active, unmuted port at unity gain.
    pub fn new(id: impl Into<String>, label: impl Into<String>, kind: IntercomPortKind) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            kind,
            active: true,
            mic_open: false,
            listen_gain_db: 0.0,
            executive_override: false,
            muted: false,
        }
    }

    /// Returns `true` if this port is currently transmitting (mic open and
    /// not muted).
    pub fn is_transmitting(&self) -> bool {
        self.mic_open && !self.muted && self.active
    }
}

impl fmt::Display for IntercomPort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = if self.active { "active" } else { "inactive" };
        write!(
            f,
            "IntercomPort('{}', {}, {})",
            self.label, self.kind, state
        )
    }
}

// ---------------------------------------------------------------------------
// PartyLine
// ---------------------------------------------------------------------------

/// A named party-line (group) channel.
///
/// All ports assigned to a party line can hear each other.  A port can be
/// a member of multiple party lines simultaneously; it hears the mix of all
/// lines it belongs to.
#[derive(Debug, Clone)]
pub struct PartyLine {
    /// Unique party-line identifier.
    pub id: String,
    /// Human-readable name (e.g. `"Camera ops"`, `"Production"`, `"A1 to A2"`).
    pub name: String,
    /// Member port ids.
    members: HashSet<String>,
    /// Whether this line is currently active (in use).
    pub active: bool,
}

impl PartyLine {
    /// Creates an empty, active party line.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            members: HashSet::new(),
            active: true,
        }
    }

    /// Adds a port to this party line.  Returns `true` if newly added.
    pub fn add_member(&mut self, port_id: impl Into<String>) -> bool {
        self.members.insert(port_id.into())
    }

    /// Removes a port from this party line.  Returns `true` if it was present.
    pub fn remove_member(&mut self, port_id: &str) -> bool {
        self.members.remove(port_id)
    }

    /// Returns `true` if the port is a member of this line.
    pub fn is_member(&self, port_id: &str) -> bool {
        self.members.contains(port_id)
    }

    /// Returns all member port ids.
    pub fn members(&self) -> impl Iterator<Item = &str> {
        self.members.iter().map(String::as_str)
    }

    /// Returns the number of members.
    pub fn member_count(&self) -> usize {
        self.members.len()
    }
}

impl fmt::Display for PartyLine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PartyLine('{}', {} members)",
            self.name,
            self.members.len()
        )
    }
}

// ---------------------------------------------------------------------------
// PointToPointCall
// ---------------------------------------------------------------------------

/// A private, direct call between exactly two ports.
///
/// A point-to-point call bypasses all party-line membership rules; only the
/// two named ports hear each other through this call.
#[derive(Debug, Clone)]
pub struct PointToPointCall {
    /// Unique call identifier.
    pub id: String,
    /// Initiating port.
    pub caller_id: String,
    /// Answering port.
    pub callee_id: String,
    /// Whether the callee has accepted the call.
    pub accepted: bool,
    /// Call state.
    pub state: CallState,
}

/// State of a point-to-point call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallState {
    /// Call has been initiated but not yet answered.
    Ringing,
    /// Call is active — both parties can communicate.
    Connected,
    /// Call has been terminated.
    Terminated,
}

impl fmt::Display for CallState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ringing => write!(f, "ringing"),
            Self::Connected => write!(f, "connected"),
            Self::Terminated => write!(f, "terminated"),
        }
    }
}

impl PointToPointCall {
    /// Creates a new call in the `Ringing` state.
    pub fn new(
        id: impl Into<String>,
        caller_id: impl Into<String>,
        callee_id: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            caller_id: caller_id.into(),
            callee_id: callee_id.into(),
            accepted: false,
            state: CallState::Ringing,
        }
    }

    /// Answers the call, moving it to `Connected`.
    pub fn answer(&mut self) {
        self.accepted = true;
        self.state = CallState::Connected;
    }

    /// Terminates the call.
    pub fn terminate(&mut self) {
        self.state = CallState::Terminated;
    }

    /// Returns `true` if either port_id matches caller or callee.
    pub fn involves(&self, port_id: &str) -> bool {
        self.caller_id == port_id || self.callee_id == port_id
    }

    /// Returns the peer of the given port within this call, if it is a
    /// participant.
    pub fn peer_of<'a>(&'a self, port_id: &str) -> Option<&'a str> {
        if self.caller_id == port_id {
            Some(&self.callee_id)
        } else if self.callee_id == port_id {
            Some(&self.caller_id)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// IntercomError
// ---------------------------------------------------------------------------

/// Errors returned by [`IntercomMatrix`] operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntercomError {
    /// A port with the given id is already registered.
    DuplicatePortId(String),
    /// No port with the given id exists.
    PortNotFound(String),
    /// A party line with the given id is already registered.
    DuplicatePartyLineId(String),
    /// No party line with the given id exists.
    PartyLineNotFound(String),
    /// A call with the given id already exists.
    DuplicateCallId(String),
    /// No call with the given id exists.
    CallNotFound(String),
    /// Cannot call a port that is already in an active call.
    PortBusy(String),
    /// Caller and callee must be different ports.
    SelfCall,
    /// The call is not in the expected state for the requested operation.
    InvalidCallState { call_id: String, state: CallState },
}

impl fmt::Display for IntercomError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicatePortId(id) => write!(f, "port already registered: {id}"),
            Self::PortNotFound(id) => write!(f, "port not found: {id}"),
            Self::DuplicatePartyLineId(id) => write!(f, "party line already registered: {id}"),
            Self::PartyLineNotFound(id) => write!(f, "party line not found: {id}"),
            Self::DuplicateCallId(id) => write!(f, "call already registered: {id}"),
            Self::CallNotFound(id) => write!(f, "call not found: {id}"),
            Self::PortBusy(id) => write!(f, "port '{id}' is already in an active call"),
            Self::SelfCall => write!(f, "caller and callee must be different ports"),
            Self::InvalidCallState { call_id, state } => {
                write!(
                    f,
                    "call '{call_id}' is in state '{state}' — cannot perform operation"
                )
            }
        }
    }
}

impl std::error::Error for IntercomError {}

// ---------------------------------------------------------------------------
// IntercomMatrix
// ---------------------------------------------------------------------------

/// Central routing authority for the intercom system.
///
/// Manages ports, party lines, and point-to-point calls, and provides
/// queries to determine which ports can hear which other ports.
#[derive(Debug, Default)]
pub struct IntercomMatrix {
    ports: HashMap<String, IntercomPort>,
    party_lines: HashMap<String, PartyLine>,
    calls: HashMap<String, PointToPointCall>,
    /// Index: port_id → set of party_line_ids the port belongs to.
    port_party_lines: HashMap<String, HashSet<String>>,
}

impl IntercomMatrix {
    /// Creates an empty intercom matrix.
    pub fn new() -> Self {
        Self::default()
    }

    // -----------------------------------------------------------------------
    // Port management
    // -----------------------------------------------------------------------

    /// Registers a new intercom port.
    pub fn register_port(&mut self, port: IntercomPort) -> Result<(), IntercomError> {
        if self.ports.contains_key(&port.id) {
            return Err(IntercomError::DuplicatePortId(port.id.clone()));
        }
        let id = port.id.clone();
        self.ports.insert(id.clone(), port);
        self.port_party_lines.entry(id).or_default();
        Ok(())
    }

    /// Removes a port and cleans up all party-line memberships and calls.
    pub fn remove_port(&mut self, port_id: &str) -> Result<IntercomPort, IntercomError> {
        let port = self
            .ports
            .remove(port_id)
            .ok_or_else(|| IntercomError::PortNotFound(port_id.to_string()))?;

        // Remove from all party lines
        if let Some(line_ids) = self.port_party_lines.remove(port_id) {
            for line_id in &line_ids {
                if let Some(line) = self.party_lines.get_mut(line_id) {
                    line.remove_member(port_id);
                }
            }
        }

        // Terminate all calls involving this port
        for call in self.calls.values_mut() {
            if call.involves(port_id) && call.state != CallState::Terminated {
                call.terminate();
            }
        }

        Ok(port)
    }

    /// Returns a reference to the port with the given id.
    pub fn get_port(&self, port_id: &str) -> Option<&IntercomPort> {
        self.ports.get(port_id)
    }

    /// Returns a mutable reference to the port with the given id.
    pub fn get_port_mut(&mut self, port_id: &str) -> Option<&mut IntercomPort> {
        self.ports.get_mut(port_id)
    }

    /// Returns all registered ports.
    pub fn all_ports(&self) -> Vec<&IntercomPort> {
        self.ports.values().collect()
    }

    // -----------------------------------------------------------------------
    // Party-line management
    // -----------------------------------------------------------------------

    /// Registers a new party line.
    pub fn register_party_line(&mut self, line: PartyLine) -> Result<(), IntercomError> {
        if self.party_lines.contains_key(&line.id) {
            return Err(IntercomError::DuplicatePartyLineId(line.id.clone()));
        }
        self.party_lines.insert(line.id.clone(), line);
        Ok(())
    }

    /// Removes a party line and removes all port memberships for it.
    pub fn remove_party_line(&mut self, line_id: &str) -> Result<PartyLine, IntercomError> {
        let line = self
            .party_lines
            .remove(line_id)
            .ok_or_else(|| IntercomError::PartyLineNotFound(line_id.to_string()))?;

        // Remove line from each member's index
        for port_id in line.members() {
            if let Some(lines) = self.port_party_lines.get_mut(port_id) {
                lines.remove(line_id);
            }
        }

        Ok(line)
    }

    /// Returns a reference to the party line with the given id.
    pub fn get_party_line(&self, line_id: &str) -> Option<&PartyLine> {
        self.party_lines.get(line_id)
    }

    /// Adds a port to a party line.
    pub fn add_port_to_party_line(
        &mut self,
        port_id: &str,
        line_id: &str,
    ) -> Result<(), IntercomError> {
        if !self.ports.contains_key(port_id) {
            return Err(IntercomError::PortNotFound(port_id.to_string()));
        }
        let line = self
            .party_lines
            .get_mut(line_id)
            .ok_or_else(|| IntercomError::PartyLineNotFound(line_id.to_string()))?;

        line.add_member(port_id);
        self.port_party_lines
            .entry(port_id.to_string())
            .or_default()
            .insert(line_id.to_string());

        Ok(())
    }

    /// Removes a port from a party line.
    pub fn remove_port_from_party_line(
        &mut self,
        port_id: &str,
        line_id: &str,
    ) -> Result<(), IntercomError> {
        let line = self
            .party_lines
            .get_mut(line_id)
            .ok_or_else(|| IntercomError::PartyLineNotFound(line_id.to_string()))?;

        line.remove_member(port_id);

        if let Some(lines) = self.port_party_lines.get_mut(port_id) {
            lines.remove(line_id);
        }

        Ok(())
    }

    /// Returns all party lines the port is a member of.
    pub fn party_lines_for_port(&self, port_id: &str) -> Vec<&PartyLine> {
        let ids = match self.port_party_lines.get(port_id) {
            Some(s) => s,
            None => return Vec::new(),
        };
        ids.iter()
            .filter_map(|id| self.party_lines.get(id))
            .collect()
    }

    // -----------------------------------------------------------------------
    // Point-to-point calls
    // -----------------------------------------------------------------------

    /// Initiates a point-to-point call between caller and callee.
    ///
    /// Fails if either port does not exist, they are the same, or the callee
    /// is already in a connected call.
    pub fn initiate_call(
        &mut self,
        call_id: impl Into<String>,
        caller_id: &str,
        callee_id: &str,
    ) -> Result<(), IntercomError> {
        if caller_id == callee_id {
            return Err(IntercomError::SelfCall);
        }
        if !self.ports.contains_key(caller_id) {
            return Err(IntercomError::PortNotFound(caller_id.to_string()));
        }
        if !self.ports.contains_key(callee_id) {
            return Err(IntercomError::PortNotFound(callee_id.to_string()));
        }

        let call_id = call_id.into();

        if self.calls.contains_key(&call_id) {
            return Err(IntercomError::DuplicateCallId(call_id));
        }

        // Check callee is not already in a connected call
        for call in self.calls.values() {
            if call.state == CallState::Connected && call.involves(callee_id) {
                return Err(IntercomError::PortBusy(callee_id.to_string()));
            }
        }

        self.calls.insert(
            call_id.clone(),
            PointToPointCall::new(call_id, caller_id, callee_id),
        );

        Ok(())
    }

    /// Answers a pending call (moves it from `Ringing` to `Connected`).
    pub fn answer_call(&mut self, call_id: &str) -> Result<(), IntercomError> {
        let call = self
            .calls
            .get_mut(call_id)
            .ok_or_else(|| IntercomError::CallNotFound(call_id.to_string()))?;

        if call.state != CallState::Ringing {
            return Err(IntercomError::InvalidCallState {
                call_id: call_id.to_string(),
                state: call.state,
            });
        }

        call.answer();
        Ok(())
    }

    /// Terminates a call (either while ringing or connected).
    pub fn terminate_call(&mut self, call_id: &str) -> Result<(), IntercomError> {
        let call = self
            .calls
            .get_mut(call_id)
            .ok_or_else(|| IntercomError::CallNotFound(call_id.to_string()))?;

        if call.state == CallState::Terminated {
            return Err(IntercomError::InvalidCallState {
                call_id: call_id.to_string(),
                state: call.state,
            });
        }

        call.terminate();
        Ok(())
    }

    /// Returns a reference to the call with the given id.
    pub fn get_call(&self, call_id: &str) -> Option<&PointToPointCall> {
        self.calls.get(call_id)
    }

    /// Returns all active (connected) calls.
    pub fn active_calls(&self) -> Vec<&PointToPointCall> {
        self.calls
            .values()
            .filter(|c| c.state == CallState::Connected)
            .collect()
    }

    // -----------------------------------------------------------------------
    // Routing queries
    // -----------------------------------------------------------------------

    /// Returns the set of port ids that can hear the given source port.
    ///
    /// A port hears another if they share at least one active party line, or
    /// if they are in a connected point-to-point call together.  A port never
    /// hears itself.
    pub fn listeners_of(&self, source_port_id: &str) -> HashSet<String> {
        let mut listeners: HashSet<String> = HashSet::new();

        // Party-line reachability
        if let Some(line_ids) = self.port_party_lines.get(source_port_id) {
            for line_id in line_ids {
                if let Some(line) = self.party_lines.get(line_id) {
                    if !line.active {
                        continue;
                    }
                    for member_id in line.members() {
                        if member_id != source_port_id {
                            listeners.insert(member_id.to_string());
                        }
                    }
                }
            }
        }

        // Point-to-point call reachability
        for call in self.calls.values() {
            if call.state == CallState::Connected {
                if let Some(peer) = call.peer_of(source_port_id) {
                    listeners.insert(peer.to_string());
                }
            }
        }

        listeners
    }

    /// Returns `true` if `listener_id` can currently hear `source_id`.
    pub fn can_hear(&self, listener_id: &str, source_id: &str) -> bool {
        self.listeners_of(source_id).contains(listener_id)
    }

    /// Returns the total number of registered ports.
    pub fn port_count(&self) -> usize {
        self.ports.len()
    }

    /// Returns the total number of registered party lines.
    pub fn party_line_count(&self) -> usize {
        self.party_lines.len()
    }

    /// Returns the total number of calls (any state).
    pub fn call_count(&self) -> usize {
        self.calls.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_port(id: &str) -> IntercomPort {
        IntercomPort::new(id, id, IntercomPortKind::Beltpack)
    }

    fn make_line(id: &str) -> PartyLine {
        PartyLine::new(id, id)
    }

    #[test]
    fn test_register_port_and_retrieve() {
        let mut matrix = IntercomMatrix::new();
        matrix.register_port(make_port("bp-1")).expect("ok");
        assert!(matrix.get_port("bp-1").is_some());
    }

    #[test]
    fn test_duplicate_port_returns_error() {
        let mut matrix = IntercomMatrix::new();
        matrix.register_port(make_port("bp-dup")).expect("ok");
        let result = matrix.register_port(make_port("bp-dup"));
        assert!(matches!(result, Err(IntercomError::DuplicatePortId(_))));
    }

    #[test]
    fn test_party_line_membership() {
        let mut matrix = IntercomMatrix::new();
        matrix.register_port(make_port("a")).expect("ok");
        matrix.register_port(make_port("b")).expect("ok");
        matrix
            .register_party_line(make_line("camera-ops"))
            .expect("ok");
        matrix
            .add_port_to_party_line("a", "camera-ops")
            .expect("ok");
        matrix
            .add_port_to_party_line("b", "camera-ops")
            .expect("ok");

        let line = matrix.get_party_line("camera-ops").expect("should exist");
        assert!(line.is_member("a"));
        assert!(line.is_member("b"));
        assert_eq!(line.member_count(), 2);
    }

    #[test]
    fn test_listeners_of_via_party_line() {
        let mut matrix = IntercomMatrix::new();
        matrix.register_port(make_port("p1")).expect("ok");
        matrix.register_port(make_port("p2")).expect("ok");
        matrix.register_port(make_port("p3")).expect("ok");
        matrix.register_party_line(make_line("line-a")).expect("ok");
        matrix.add_port_to_party_line("p1", "line-a").expect("ok");
        matrix.add_port_to_party_line("p2", "line-a").expect("ok");

        let listeners = matrix.listeners_of("p1");
        assert!(listeners.contains("p2"));
        assert!(!listeners.contains("p3"));
        assert!(!listeners.contains("p1")); // self not included
    }

    #[test]
    fn test_point_to_point_call_flow() {
        let mut matrix = IntercomMatrix::new();
        matrix.register_port(make_port("caller")).expect("ok");
        matrix.register_port(make_port("callee")).expect("ok");

        matrix
            .initiate_call("call-1", "caller", "callee")
            .expect("ok");
        let call = matrix.get_call("call-1").expect("should exist");
        assert_eq!(call.state, CallState::Ringing);

        matrix.answer_call("call-1").expect("ok");
        let call = matrix.get_call("call-1").expect("should exist");
        assert_eq!(call.state, CallState::Connected);
    }

    #[test]
    fn test_can_hear_after_call_connected() {
        let mut matrix = IntercomMatrix::new();
        matrix.register_port(make_port("dir")).expect("ok");
        matrix.register_port(make_port("prod")).expect("ok");

        matrix.initiate_call("call-2", "dir", "prod").expect("ok");
        assert!(!matrix.can_hear("prod", "dir")); // not yet connected
        matrix.answer_call("call-2").expect("ok");
        assert!(matrix.can_hear("prod", "dir"));
        assert!(matrix.can_hear("dir", "prod"));
    }

    #[test]
    fn test_terminate_call() {
        let mut matrix = IntercomMatrix::new();
        matrix.register_port(make_port("x")).expect("ok");
        matrix.register_port(make_port("y")).expect("ok");
        matrix.initiate_call("call-3", "x", "y").expect("ok");
        matrix.answer_call("call-3").expect("ok");
        matrix.terminate_call("call-3").expect("ok");
        let call = matrix.get_call("call-3").expect("should exist");
        assert_eq!(call.state, CallState::Terminated);
        assert!(!matrix.can_hear("y", "x"));
    }

    #[test]
    fn test_self_call_returns_error() {
        let mut matrix = IntercomMatrix::new();
        matrix.register_port(make_port("solo")).expect("ok");
        let result = matrix.initiate_call("call-self", "solo", "solo");
        assert!(matches!(result, Err(IntercomError::SelfCall)));
    }

    #[test]
    fn test_port_busy_blocks_second_call() {
        let mut matrix = IntercomMatrix::new();
        matrix.register_port(make_port("a1")).expect("ok");
        matrix.register_port(make_port("b1")).expect("ok");
        matrix.register_port(make_port("c1")).expect("ok");
        matrix.initiate_call("call-ab", "a1", "b1").expect("ok");
        matrix.answer_call("call-ab").expect("ok");

        // c1 tries to call b1 while b1 is in a connected call
        let result = matrix.initiate_call("call-cb", "c1", "b1");
        assert!(matches!(result, Err(IntercomError::PortBusy(_))));
    }

    #[test]
    fn test_remove_port_cleans_party_lines() {
        let mut matrix = IntercomMatrix::new();
        matrix.register_port(make_port("rm-a")).expect("ok");
        matrix.register_port(make_port("rm-b")).expect("ok");
        matrix
            .register_party_line(make_line("line-rm"))
            .expect("ok");
        matrix
            .add_port_to_party_line("rm-a", "line-rm")
            .expect("ok");
        matrix
            .add_port_to_party_line("rm-b", "line-rm")
            .expect("ok");

        matrix.remove_port("rm-a").expect("ok");

        let line = matrix.get_party_line("line-rm").expect("should exist");
        assert!(!line.is_member("rm-a"));
        assert!(line.is_member("rm-b"));
    }

    #[test]
    fn test_remove_port_terminates_calls() {
        let mut matrix = IntercomMatrix::new();
        matrix.register_port(make_port("trm-a")).expect("ok");
        matrix.register_port(make_port("trm-b")).expect("ok");
        matrix
            .initiate_call("call-trm", "trm-a", "trm-b")
            .expect("ok");
        matrix.answer_call("call-trm").expect("ok");

        matrix.remove_port("trm-a").expect("ok");

        let call = matrix.get_call("call-trm").expect("should exist");
        assert_eq!(call.state, CallState::Terminated);
    }

    #[test]
    fn test_party_lines_for_port() {
        let mut matrix = IntercomMatrix::new();
        matrix.register_port(make_port("multi")).expect("ok");
        matrix.register_party_line(make_line("line-x")).expect("ok");
        matrix.register_party_line(make_line("line-y")).expect("ok");
        matrix
            .add_port_to_party_line("multi", "line-x")
            .expect("ok");
        matrix
            .add_port_to_party_line("multi", "line-y")
            .expect("ok");

        let lines = matrix.party_lines_for_port("multi");
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn test_cross_line_isolation() {
        // Ports on different party lines should NOT hear each other
        let mut matrix = IntercomMatrix::new();
        matrix.register_port(make_port("alpha")).expect("ok");
        matrix.register_port(make_port("beta")).expect("ok");
        matrix.register_port(make_port("gamma")).expect("ok");
        matrix.register_party_line(make_line("line-1")).expect("ok");
        matrix.register_party_line(make_line("line-2")).expect("ok");
        matrix
            .add_port_to_party_line("alpha", "line-1")
            .expect("ok");
        matrix.add_port_to_party_line("beta", "line-1").expect("ok");
        matrix
            .add_port_to_party_line("gamma", "line-2")
            .expect("ok");

        assert!(matrix.can_hear("beta", "alpha")); // same line
        assert!(!matrix.can_hear("gamma", "alpha")); // different lines
    }

    #[test]
    fn test_display_implementations() {
        let port = make_port("test");
        let s = format!("{port}");
        assert!(s.contains("test"));

        let line = make_line("production");
        let s = format!("{line}");
        assert!(s.contains("production"));
    }

    // -----------------------------------------------------------------------
    // Additional tests (8+) for full coverage
    // -----------------------------------------------------------------------

    #[test]
    fn test_port_transmitting_state() {
        let mut port = IntercomPort::new("bp", "beltpack", IntercomPortKind::Beltpack);
        assert!(!port.is_transmitting()); // mic not open
        port.mic_open = true;
        assert!(port.is_transmitting()); // now transmitting
        port.muted = true;
        assert!(!port.is_transmitting()); // muted
        port.muted = false;
        port.active = false;
        assert!(!port.is_transmitting()); // inactive
    }

    #[test]
    fn test_port_count_and_party_line_count() {
        let mut matrix = IntercomMatrix::new();
        matrix.register_port(make_port("p1")).expect("ok");
        matrix.register_port(make_port("p2")).expect("ok");
        matrix
            .register_party_line(make_line("line-count"))
            .expect("ok");
        assert_eq!(matrix.port_count(), 2);
        assert_eq!(matrix.party_line_count(), 1);
        assert_eq!(matrix.call_count(), 0);
    }

    #[test]
    fn test_remove_party_line_cleans_index() {
        let mut matrix = IntercomMatrix::new();
        matrix.register_port(make_port("p")).expect("ok");
        matrix
            .register_party_line(make_line("line-rm"))
            .expect("ok");
        matrix.add_port_to_party_line("p", "line-rm").expect("ok");

        matrix.remove_party_line("line-rm").expect("ok");

        let lines = matrix.party_lines_for_port("p");
        assert!(lines.is_empty());
    }

    #[test]
    fn test_remove_port_from_party_line() {
        let mut matrix = IntercomMatrix::new();
        matrix.register_port(make_port("pa")).expect("ok");
        matrix.register_port(make_port("pb")).expect("ok");
        matrix
            .register_party_line(make_line("line-remove"))
            .expect("ok");
        matrix
            .add_port_to_party_line("pa", "line-remove")
            .expect("ok");
        matrix
            .add_port_to_party_line("pb", "line-remove")
            .expect("ok");
        matrix
            .remove_port_from_party_line("pa", "line-remove")
            .expect("ok");

        let line = matrix.get_party_line("line-remove").expect("should exist");
        assert!(!line.is_member("pa"));
        assert!(line.is_member("pb"));
    }

    #[test]
    fn test_active_calls_filter() {
        let mut matrix = IntercomMatrix::new();
        matrix.register_port(make_port("u1")).expect("ok");
        matrix.register_port(make_port("u2")).expect("ok");
        matrix.register_port(make_port("u3")).expect("ok");
        matrix
            .initiate_call("ringing-call", "u1", "u2")
            .expect("ok");
        matrix
            .initiate_call("connected-call", "u1", "u3")
            .expect("ok");
        matrix.answer_call("connected-call").expect("ok");

        let active = matrix.active_calls();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, "connected-call");
    }

    #[test]
    fn test_answer_already_connected_call_returns_error() {
        let mut matrix = IntercomMatrix::new();
        matrix.register_port(make_port("ca")).expect("ok");
        matrix.register_port(make_port("cb")).expect("ok");
        matrix.initiate_call("call-a", "ca", "cb").expect("ok");
        matrix.answer_call("call-a").expect("ok");
        let result = matrix.answer_call("call-a");
        assert!(matches!(
            result,
            Err(IntercomError::InvalidCallState { .. })
        ));
    }

    #[test]
    fn test_terminate_already_terminated_returns_error() {
        let mut matrix = IntercomMatrix::new();
        matrix.register_port(make_port("tx")).expect("ok");
        matrix.register_port(make_port("ty")).expect("ok");
        matrix.initiate_call("call-t", "tx", "ty").expect("ok");
        matrix.terminate_call("call-t").expect("ok");
        let result = matrix.terminate_call("call-t");
        assert!(matches!(
            result,
            Err(IntercomError::InvalidCallState { .. })
        ));
    }

    #[test]
    fn test_port_not_found_for_call_initiation() {
        let mut matrix = IntercomMatrix::new();
        matrix.register_port(make_port("real-port")).expect("ok");
        let result = matrix.initiate_call("c1", "real-port", "ghost-port");
        assert!(matches!(result, Err(IntercomError::PortNotFound(_))));
    }

    #[test]
    fn test_party_line_inactive_hides_from_listeners() {
        let mut matrix = IntercomMatrix::new();
        matrix.register_port(make_port("n1")).expect("ok");
        matrix.register_port(make_port("n2")).expect("ok");
        let mut line = PartyLine::new("inactive-line", "inactive");
        line.active = false;
        matrix.register_party_line(line).expect("ok");
        matrix
            .add_port_to_party_line("n1", "inactive-line")
            .expect("ok");
        matrix
            .add_port_to_party_line("n2", "inactive-line")
            .expect("ok");

        // Since line is inactive, n2 should not hear n1
        assert!(!matrix.can_hear("n2", "n1"));
    }

    #[test]
    fn test_duplicate_party_line_returns_error() {
        let mut matrix = IntercomMatrix::new();
        matrix
            .register_party_line(make_line("dup-line"))
            .expect("ok");
        let result = matrix.register_party_line(make_line("dup-line"));
        assert!(matches!(
            result,
            Err(IntercomError::DuplicatePartyLineId(_))
        ));
    }

    #[test]
    fn test_get_port_mut_allows_modification() {
        let mut matrix = IntercomMatrix::new();
        matrix.register_port(make_port("mutable-port")).expect("ok");
        if let Some(port) = matrix.get_port_mut("mutable-port") {
            port.listen_gain_db = 6.0;
        }
        let port = matrix.get_port("mutable-port").expect("should exist");
        assert!((port.listen_gain_db - 6.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_call_peer_of() {
        let call = PointToPointCall::new("c", "alice", "bob");
        assert_eq!(call.peer_of("alice"), Some("bob"));
        assert_eq!(call.peer_of("bob"), Some("alice"));
        assert_eq!(call.peer_of("charlie"), None);
    }

    #[test]
    fn test_call_involves() {
        let call = PointToPointCall::new("c", "alice", "bob");
        assert!(call.involves("alice"));
        assert!(call.involves("bob"));
        assert!(!call.involves("charlie"));
    }

    #[test]
    fn test_intercom_error_display() {
        let err = IntercomError::PortNotFound("bp-99".to_string());
        assert!(format!("{err}").contains("bp-99"));
        let err2 = IntercomError::SelfCall;
        assert!(format!("{err2}").contains("different"));
    }

    #[test]
    fn test_all_ports_returns_all() {
        let mut matrix = IntercomMatrix::new();
        matrix.register_port(make_port("x1")).expect("ok");
        matrix.register_port(make_port("x2")).expect("ok");
        matrix.register_port(make_port("x3")).expect("ok");
        assert_eq!(matrix.all_ports().len(), 3);
    }

    #[test]
    fn test_party_line_member_count() {
        let mut line = PartyLine::new("pl", "party");
        assert_eq!(line.member_count(), 0);
        line.add_member("a");
        line.add_member("b");
        assert_eq!(line.member_count(), 2);
        line.remove_member("a");
        assert_eq!(line.member_count(), 1);
    }

    #[test]
    fn test_ringing_call_does_not_block_new_call_to_same_callee() {
        // Only a CONNECTED call should block; a ringing call should not.
        let mut matrix = IntercomMatrix::new();
        matrix.register_port(make_port("caller-a")).expect("ok");
        matrix.register_port(make_port("caller-b")).expect("ok");
        matrix.register_port(make_port("callee-z")).expect("ok");
        matrix
            .initiate_call("ring-call", "caller-a", "callee-z")
            .expect("ok");
        // callee-z is ringing (not connected), so caller-b should be able to call them
        assert!(matrix
            .initiate_call("new-call", "caller-b", "callee-z")
            .is_ok());
    }
}
