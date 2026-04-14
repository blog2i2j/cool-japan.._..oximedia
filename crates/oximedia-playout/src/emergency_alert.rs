//! Emergency Alert System (EAS) support for broadcast playout.
//!
//! This module implements:
//!
//! - **EAS header parsing** — decoding of SAME (Specific Area Message Encoding)
//!   headers as used in US EAS and compatible systems (EAN, CAP-like structures)
//! - **Alert priority model** — multi-level urgency classification from
//!   informational notices to life-safety emergencies
//! - **Interrupt scheduler** — frame-accurate insertion of alert content into
//!   the playout timeline, including pre-emption of normal content
//! - **Acknowledgement tracking** — per-alert delivery confirmation for
//!   redundant transmission paths
//!
//! # SAME header format (abbreviated)
//!
//! ```text
//! ZCZC-ORG-EEE-PSSCCC+TTTT-JJJHHMM-LLLLLLLL-
//! ```
//!
//! Where:
//! - `ORG`  — originator code (`EAS`, `CIV`, `WXR`, `PEP`)
//! - `EEE`  — event code (e.g. `EAN`, `TOR`, `SVR`, `EVI`)
//! - `PSSCCC` — FIPS geographic code (repeated up to 31 times)
//! - `TTTT` — purge time (hours + minutes, e.g. `0130`)
//! - `JJJHHMM` — issue time (Julian day + HH:MM UTC)
//! - `LLLLLLLL` — identification of issuing station
//!
//! # References
//! - FCC 47 CFR Part 11 (Emergency Alert System)
//! - FEMA IPAWS SAME standard

#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};
use std::fmt;

// ── Alert priority ─────────────────────────────────────────────────────────────

/// Urgency / priority level of an emergency alert.
///
/// Levels are ordered from lowest (informational) to highest (life-safety).
/// The numeric discriminants can be compared directly with `<` / `>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AlertPriority {
    /// Administrative test or periodic weekly test.
    Test = 0,
    /// Non-emergency informational announcement.
    Statement = 1,
    /// Advisory: minor impacts possible, awareness recommended.
    Advisory = 2,
    /// Watch: conditions are favourable for a hazard but not yet imminent.
    Watch = 3,
    /// Warning: hazardous conditions are occurring or imminent.
    Warning = 4,
    /// Emergency: immediate life-safety threat.
    Emergency = 5,
    /// Presidential / national-level emergency (highest priority).
    Presidential = 6,
}

impl AlertPriority {
    /// Returns `true` if this priority level requires immediate interruption
    /// of normal playout.
    pub fn requires_interrupt(&self) -> bool {
        matches!(self, Self::Warning | Self::Emergency | Self::Presidential)
    }

    /// Return a human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Test => "TEST",
            Self::Statement => "STATEMENT",
            Self::Watch => "WATCH",
            Self::Advisory => "ADVISORY",
            Self::Warning => "WARNING",
            Self::Emergency => "EMERGENCY",
            Self::Presidential => "PRESIDENTIAL",
        }
    }
}

impl fmt::Display for AlertPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

// ── EAS event codes ────────────────────────────────────────────────────────────

/// Decoded EAS event code with priority mapping.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EasEventCode {
    /// Three-letter code as specified in FCC Part 11 / FEMA IPAWS.
    pub code: String,
    /// Human-readable description of this event type.
    pub description: String,
    /// Mapped priority level.
    pub priority: AlertPriority,
}

impl EasEventCode {
    /// Look up a known event code.
    ///
    /// Returns `None` if the code is not in the built-in table; callers can
    /// create custom codes directly via [`EasEventCode`] struct construction.
    pub fn lookup(code: &str) -> Option<Self> {
        KNOWN_EVENT_CODES.iter().find(|(c, _, _)| *c == code).map(|(c, d, p)| Self {
            code: c.to_string(),
            description: d.to_string(),
            priority: *p,
        })
    }

    /// Return an unknown/custom event code mapped to `Advisory` priority.
    pub fn unknown(code: &str) -> Self {
        Self {
            code: code.to_string(),
            description: format!("Unknown event code: {code}"),
            priority: AlertPriority::Advisory,
        }
    }
}

/// Static table of known EAS event codes (FCC Part 11 / FEMA).
static KNOWN_EVENT_CODES: &[(&str, &str, AlertPriority)] = &[
    ("EAN", "Emergency Action Notification (Presidential)", AlertPriority::Presidential),
    ("EAT", "Emergency Action Termination", AlertPriority::Presidential),
    ("NIC", "National Information Center", AlertPriority::Presidential),
    ("NPT", "National Periodic Test", AlertPriority::Test),
    ("RMT", "Required Monthly Test", AlertPriority::Test),
    ("RWT", "Required Weekly Test", AlertPriority::Test),
    ("ADR", "Administrative Message", AlertPriority::Statement),
    ("AVA", "Avalanche Watch", AlertPriority::Watch),
    ("AVW", "Avalanche Warning", AlertPriority::Warning),
    ("BZW", "Blizzard Warning", AlertPriority::Warning),
    ("CAE", "Child Abduction Emergency", AlertPriority::Emergency),
    ("CDW", "Civil Danger Warning", AlertPriority::Warning),
    ("CEM", "Civil Emergency Message", AlertPriority::Emergency),
    ("EQW", "Earthquake Warning", AlertPriority::Warning),
    ("EVI", "Evacuation Immediate", AlertPriority::Emergency),
    ("FRW", "Fire Warning", AlertPriority::Warning),
    ("HMW", "Hazardous Materials Warning", AlertPriority::Warning),
    ("LAE", "Local Area Emergency", AlertPriority::Emergency),
    ("LEW", "Law Enforcement Warning", AlertPriority::Warning),
    ("NUW", "Nuclear Power Plant Warning", AlertPriority::Warning),
    ("RHW", "Radiological Hazard Warning", AlertPriority::Warning),
    ("SPW", "Shelter in Place Warning", AlertPriority::Warning),
    ("SVA", "Severe Thunderstorm Watch", AlertPriority::Watch),
    ("SVR", "Severe Thunderstorm Warning", AlertPriority::Warning),
    ("SVS", "Severe Weather Statement", AlertPriority::Statement),
    ("TOA", "Tornado Watch", AlertPriority::Watch),
    ("TOR", "Tornado Warning", AlertPriority::Warning),
    ("TRA", "Tropical Storm Watch", AlertPriority::Watch),
    ("TRW", "Tropical Storm Warning", AlertPriority::Warning),
    ("TSA", "Tsunami Watch", AlertPriority::Watch),
    ("TSW", "Tsunami Warning", AlertPriority::Warning),
    ("VOW", "Volcano Warning", AlertPriority::Warning),
    ("WSA", "Winter Storm Watch", AlertPriority::Watch),
    ("WSW", "Winter Storm Warning", AlertPriority::Warning),
];

// ── SAME header parser ─────────────────────────────────────────────────────────

/// Parsed representation of an EAS SAME (Specific Area Message Encoding) header.
#[derive(Debug, Clone, PartialEq)]
pub struct SameHeader {
    /// Originator code (`EAS`, `CIV`, `WXR`, `PEP`).
    pub originator: String,
    /// Decoded event code with priority.
    pub event: EasEventCode,
    /// FIPS location codes (up to 31).
    pub locations: Vec<String>,
    /// Purge time: hours component (0–99).
    pub purge_hours: u8,
    /// Purge time: minutes component (0–59).
    pub purge_minutes: u8,
    /// Issue time: Julian day of year (1–366).
    pub issue_julian_day: u16,
    /// Issue time: hour (0–23, UTC).
    pub issue_hour: u8,
    /// Issue time: minute (0–59, UTC).
    pub issue_minute: u8,
    /// Identification of the originating station (up to 8 chars).
    pub station_id: String,
}

impl SameHeader {
    /// Parse a SAME header string.
    ///
    /// Accepts headers starting with `ZCZC-` (activation) or strips the
    /// leading `ZCZC-` prefix if absent.  The trailing `-` and any CR/LF are
    /// tolerated.
    ///
    /// # Errors
    ///
    /// Returns [`ParseError`] if the header cannot be decoded.
    pub fn parse(input: &str) -> Result<Self, ParseError> {
        let s = input.trim().trim_end_matches('-');

        // Strip optional ZCZC- prefix
        let body = if let Some(stripped) = s.strip_prefix("ZCZC-") {
            stripped
        } else {
            s
        };

        // Split on '-'
        let parts: Vec<&str> = body.splitn(6, '-').collect();
        if parts.len() < 5 {
            return Err(ParseError::TooFewFields {
                expected: 5,
                found: parts.len(),
            });
        }

        let originator = parts[0].to_string();
        let event_code_str = parts[1];
        let event = EasEventCode::lookup(event_code_str)
            .unwrap_or_else(|| EasEventCode::unknown(event_code_str));

        // Locations + purge time are packed together in parts[2] as
        // "LOC+TTTT" with optional additional "LOC+TTTT" tokens separated by '-'
        // In standard SAME: part[2] = PSSCCC+TTTT (first location includes purge)
        // Additional location codes are inserted as separate '-' segments before the purge.
        // We join all middle fields and re-parse.
        let geo_and_time_fields: Vec<&str> = parts[2..parts.len() - 2].to_vec();

        let mut locations: Vec<String> = Vec::new();
        let mut purge_hours: u8 = 0;
        let mut purge_minutes: u8 = 0;

        for field in &geo_and_time_fields {
            // Each field is either "PSSCCC" (pure location) or "PSSCCC+TTTT" (last one)
            if let Some(plus_pos) = field.find('+') {
                locations.push(field[..plus_pos].to_string());
                let purge_str = &field[plus_pos + 1..];
                if purge_str.len() == 4 {
                    purge_hours = purge_str[..2]
                        .parse::<u8>()
                        .map_err(|_| ParseError::InvalidPurgeTime(purge_str.to_string()))?;
                    purge_minutes = purge_str[2..]
                        .parse::<u8>()
                        .map_err(|_| ParseError::InvalidPurgeTime(purge_str.to_string()))?;
                } else {
                    return Err(ParseError::InvalidPurgeTime(purge_str.to_string()));
                }
            } else {
                locations.push(field.to_string());
            }
        }

        if locations.is_empty() {
            return Err(ParseError::NoLocationCodes);
        }

        // Issue time: "JJJHHMM"
        let issue_str = parts[parts.len() - 2];
        if issue_str.len() < 7 {
            return Err(ParseError::InvalidIssueTime(issue_str.to_string()));
        }
        let issue_julian_day = issue_str[..3]
            .parse::<u16>()
            .map_err(|_| ParseError::InvalidIssueTime(issue_str.to_string()))?;
        let issue_hour = issue_str[3..5]
            .parse::<u8>()
            .map_err(|_| ParseError::InvalidIssueTime(issue_str.to_string()))?;
        let issue_minute = issue_str[5..7]
            .parse::<u8>()
            .map_err(|_| ParseError::InvalidIssueTime(issue_str.to_string()))?;

        let station_id = parts[parts.len() - 1].to_string();

        Ok(Self {
            originator,
            event,
            locations,
            purge_hours,
            purge_minutes,
            issue_julian_day,
            issue_hour,
            issue_minute,
            station_id,
        })
    }

    /// Return the total purge duration in seconds.
    pub fn purge_duration_secs(&self) -> u32 {
        (self.purge_hours as u32) * 3600 + (self.purge_minutes as u32) * 60
    }

    /// Return `true` if this header represents a test message.
    pub fn is_test(&self) -> bool {
        self.event.priority == AlertPriority::Test
    }
}

// ── Parse errors ──────────────────────────────────────────────────────────────

/// Errors returned by the SAME header parser.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    /// Header has fewer fields than expected.
    TooFewFields { expected: usize, found: usize },
    /// The purge time field could not be decoded.
    InvalidPurgeTime(String),
    /// The issue time field could not be decoded.
    InvalidIssueTime(String),
    /// No geographic location codes were found.
    NoLocationCodes,
    /// The input string was empty.
    EmptyInput,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewFields { expected, found } => {
                write!(f, "SAME header has {found} fields; expected at least {expected}")
            }
            Self::InvalidPurgeTime(s) => write!(f, "invalid purge time: '{s}'"),
            Self::InvalidIssueTime(s) => write!(f, "invalid issue time: '{s}'"),
            Self::NoLocationCodes => write!(f, "no geographic location codes in header"),
            Self::EmptyInput => write!(f, "empty SAME header input"),
        }
    }
}

impl std::error::Error for ParseError {}

// ── Alert record ──────────────────────────────────────────────────────────────

/// Unique identifier for a scheduled alert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AlertId(pub u64);

impl fmt::Display for AlertId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ALERT-{:06}", self.0)
    }
}

/// Status of an alert in the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertStatus {
    /// Alert has been received and is queued but not yet aired.
    Pending,
    /// Alert is currently airing.
    Active,
    /// Alert has completed and is awaiting acknowledgement.
    PendingAck,
    /// Alert has been fully acknowledged and is complete.
    Acknowledged,
    /// Alert was cancelled before it aired.
    Cancelled,
    /// Alert expired (purge time elapsed) before being acknowledged.
    Expired,
}

/// A complete alert record, combining the parsed SAME header with scheduling
/// metadata.
#[derive(Debug, Clone)]
pub struct AlertRecord {
    /// Unique identifier assigned by the scheduler.
    pub id: AlertId,
    /// Parsed SAME header.
    pub header: SameHeader,
    /// Current lifecycle status.
    pub status: AlertStatus,
    /// Frame at which the alert is scheduled to begin.
    pub scheduled_frame: u64,
    /// Frame at which the alert actually started (set when status → Active).
    pub actual_start_frame: Option<u64>,
    /// Frame at which the alert ended.
    pub actual_end_frame: Option<u64>,
    /// Number of times this alert has been transmitted (for redundancy).
    pub transmission_count: u32,
    /// Acknowledgement receipts from downstream relays / stations.
    pub acks: Vec<AckReceipt>,
}

/// Acknowledgement receipt from a downstream relay or station.
#[derive(Debug, Clone, PartialEq)]
pub struct AckReceipt {
    /// Identifying string for the receiving station.
    pub station_id: String,
    /// Frame at which the ack was received.
    pub frame: u64,
}

// ── Interrupt scheduler ────────────────────────────────────────────────────────

/// Errors from the alert scheduler.
#[derive(Debug, Clone, PartialEq)]
pub enum AlertSchedulerError {
    /// An alert with the same ID already exists.
    DuplicateAlertId(AlertId),
    /// The alert was not found.
    AlertNotFound(AlertId),
    /// The alert is in a state that does not permit the requested operation.
    InvalidStateTransition { from: AlertStatus, to: AlertStatus },
    /// Minimum priority filter prevented scheduling.
    BelowMinimumPriority { got: AlertPriority, min: AlertPriority },
}

impl fmt::Display for AlertSchedulerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateAlertId(id) => write!(f, "alert {id} already scheduled"),
            Self::AlertNotFound(id) => write!(f, "alert {id} not found"),
            Self::InvalidStateTransition { from, to } => {
                write!(f, "invalid state transition {from:?} → {to:?}")
            }
            Self::BelowMinimumPriority { got, min } => {
                write!(f, "alert priority {got} is below minimum {min}")
            }
        }
    }
}

impl std::error::Error for AlertSchedulerError {}

/// Configuration for the alert interrupt scheduler.
#[derive(Debug, Clone)]
pub struct AlertSchedulerConfig {
    /// Minimum priority level that will be admitted to the schedule.
    /// Alerts below this level are silently dropped.
    pub min_priority: AlertPriority,
    /// Frame-level look-ahead: how many frames in advance an alert is prepared.
    pub lookahead_frames: u64,
    /// Maximum number of concurrent active alerts.
    pub max_concurrent: usize,
    /// Whether test messages are admitted even if below `min_priority`.
    pub admit_tests: bool,
}

impl Default for AlertSchedulerConfig {
    fn default() -> Self {
        Self {
            min_priority: AlertPriority::Advisory,
            lookahead_frames: 25, // 1 second at 25fps
            max_concurrent: 3,
            admit_tests: true,
        }
    }
}

/// Frame-accurate interrupt scheduler for emergency alerts.
///
/// Maintains a priority-ordered queue of [`AlertRecord`]s and generates
/// interrupt cue events that can be acted upon by the playout engine.
#[derive(Debug)]
pub struct AlertScheduler {
    /// Scheduler configuration.
    config: AlertSchedulerConfig,
    /// All known alerts, keyed by ID.
    alerts: HashMap<AlertId, AlertRecord>,
    /// Pending queue (sorted by priority desc, then scheduled frame asc).
    pending_queue: VecDeque<AlertId>,
    /// Currently active alert IDs (ordered by start frame).
    active_alerts: Vec<AlertId>,
    /// Monotonic ID counter.
    next_id: u64,
    /// Current frame position.
    current_frame: u64,
    /// Log of interrupt events.
    interrupt_log: Vec<InterruptEvent>,
}

/// Type of interrupt event emitted by the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterruptEventKind {
    /// Normal playout should be pre-empted; alert content should begin.
    Begin,
    /// Alert content has completed; normal playout may resume.
    End,
    /// Alert has been retransmitted (redundancy repeat).
    Retransmit,
    /// Alert expired without full acknowledgement.
    Expire,
}

/// An interrupt event produced by the scheduler.
#[derive(Debug, Clone)]
pub struct InterruptEvent {
    pub alert_id: AlertId,
    pub kind: InterruptEventKind,
    pub frame: u64,
    pub priority: AlertPriority,
}

impl AlertScheduler {
    /// Create a new scheduler with the given configuration.
    pub fn new(config: AlertSchedulerConfig) -> Self {
        Self {
            config,
            alerts: HashMap::new(),
            pending_queue: VecDeque::new(),
            active_alerts: Vec::new(),
            next_id: 1,
            current_frame: 0,
            interrupt_log: Vec::new(),
        }
    }

    /// Allocate a new unique [`AlertId`].
    fn alloc_id(&mut self) -> AlertId {
        let id = AlertId(self.next_id);
        self.next_id = self.next_id.saturating_add(1);
        id
    }

    // ── Intake ────────────────────────────────────────────────────────────────

    /// Admit a new alert from a parsed [`SameHeader`].
    ///
    /// The alert is inserted into the priority queue. Returns the assigned
    /// [`AlertId`].
    pub fn admit(&mut self, header: SameHeader, scheduled_frame: u64) -> Result<AlertId, AlertSchedulerError> {
        // Priority filter
        let passes = header.event.priority >= self.config.min_priority
            || (self.config.admit_tests && header.is_test());
        if !passes {
            return Err(AlertSchedulerError::BelowMinimumPriority {
                got: header.event.priority,
                min: self.config.min_priority,
            });
        }

        let id = self.alloc_id();
        let record = AlertRecord {
            id,
            header,
            status: AlertStatus::Pending,
            scheduled_frame,
            actual_start_frame: None,
            actual_end_frame: None,
            transmission_count: 0,
            acks: Vec::new(),
        };

        self.alerts.insert(id, record);
        self.enqueue_pending(id);
        Ok(id)
    }

    /// Insert `id` into `pending_queue` in priority-descending order.
    fn enqueue_pending(&mut self, id: AlertId) {
        let priority = self
            .alerts
            .get(&id)
            .map(|r| r.header.event.priority)
            .unwrap_or(AlertPriority::Statement);
        let sched_frame = self
            .alerts
            .get(&id)
            .map(|r| r.scheduled_frame)
            .unwrap_or(0);

        // Find insertion point: higher priority first, then earlier frame
        let pos = self.pending_queue.iter().position(|&eid| {
            let ep = self.alerts.get(&eid).map(|r| r.header.event.priority).unwrap_or(AlertPriority::Statement);
            let ef = self.alerts.get(&eid).map(|r| r.scheduled_frame).unwrap_or(0);
            // Current entry eid should come after id if id has higher priority,
            // or same priority and earlier frame.
            (priority, std::cmp::Reverse(sched_frame)) > (ep, std::cmp::Reverse(ef))
        });

        match pos {
            Some(p) => self.pending_queue.insert(p, id),
            None => self.pending_queue.push_back(id),
        }
    }

    // ── Frame advance ─────────────────────────────────────────────────────────

    /// Advance the scheduler to `target_frame`.
    ///
    /// Returns all interrupt events that fire between the previous cursor
    /// position and `target_frame` inclusive.
    pub fn advance_to_frame(&mut self, target_frame: u64) -> Vec<InterruptEvent> {
        let mut events = Vec::new();

        // Activate pending alerts whose scheduled_frame <= target_frame
        // respecting max_concurrent.
        let mut newly_active: Vec<AlertId> = Vec::new();
        for &id in &self.pending_queue {
            if self.active_alerts.len() + newly_active.len() >= self.config.max_concurrent {
                break;
            }
            if let Some(rec) = self.alerts.get(&id) {
                if rec.scheduled_frame <= target_frame
                    && rec.status == AlertStatus::Pending
                {
                    newly_active.push(id);
                }
            }
        }

        for id in newly_active {
            self.pending_queue.retain(|&eid| eid != id);
            if let Some(rec) = self.alerts.get_mut(&id) {
                rec.status = AlertStatus::Active;
                rec.actual_start_frame = Some(target_frame);
                rec.transmission_count += 1;
                let ev = InterruptEvent {
                    alert_id: id,
                    kind: InterruptEventKind::Begin,
                    frame: target_frame,
                    priority: rec.header.event.priority,
                };
                self.interrupt_log.push(ev.clone());
                events.push(ev);
                self.active_alerts.push(id);
            }
        }

        self.current_frame = target_frame;
        events
    }

    // ── State transitions ─────────────────────────────────────────────────────

    /// Mark an active alert as completed (content has finished airing).
    pub fn complete_alert(&mut self, id: AlertId) -> Result<(), AlertSchedulerError> {
        let rec = self.alerts.get_mut(&id).ok_or(AlertSchedulerError::AlertNotFound(id))?;
        if rec.status != AlertStatus::Active {
            return Err(AlertSchedulerError::InvalidStateTransition {
                from: rec.status,
                to: AlertStatus::PendingAck,
            });
        }
        rec.status = AlertStatus::PendingAck;
        rec.actual_end_frame = Some(self.current_frame);
        self.active_alerts.retain(|&eid| eid != id);
        let ev = InterruptEvent {
            alert_id: id,
            kind: InterruptEventKind::End,
            frame: self.current_frame,
            priority: rec.header.event.priority,
        };
        self.interrupt_log.push(ev);
        Ok(())
    }

    /// Record an acknowledgement receipt for an alert.
    pub fn acknowledge(
        &mut self,
        id: AlertId,
        station_id: &str,
    ) -> Result<(), AlertSchedulerError> {
        let rec = self.alerts.get_mut(&id).ok_or(AlertSchedulerError::AlertNotFound(id))?;
        if !matches!(rec.status, AlertStatus::PendingAck | AlertStatus::Active) {
            return Err(AlertSchedulerError::InvalidStateTransition {
                from: rec.status,
                to: AlertStatus::Acknowledged,
            });
        }
        rec.acks.push(AckReceipt {
            station_id: station_id.to_string(),
            frame: self.current_frame,
        });
        rec.status = AlertStatus::Acknowledged;
        Ok(())
    }

    /// Cancel a pending alert before it has aired.
    pub fn cancel(&mut self, id: AlertId) -> Result<(), AlertSchedulerError> {
        let rec = self.alerts.get_mut(&id).ok_or(AlertSchedulerError::AlertNotFound(id))?;
        if rec.status != AlertStatus::Pending {
            return Err(AlertSchedulerError::InvalidStateTransition {
                from: rec.status,
                to: AlertStatus::Cancelled,
            });
        }
        rec.status = AlertStatus::Cancelled;
        self.pending_queue.retain(|&eid| eid != id);
        Ok(())
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// Return the record for `id`, if it exists.
    pub fn record(&self, id: AlertId) -> Option<&AlertRecord> {
        self.alerts.get(&id)
    }

    /// Return IDs of all alerts currently active (airing).
    pub fn active_alert_ids(&self) -> &[AlertId] {
        &self.active_alerts
    }

    /// Return IDs of all pending (queued) alerts.
    pub fn pending_ids(&self) -> Vec<AlertId> {
        self.pending_queue.iter().copied().collect()
    }

    /// Return the highest priority among all pending and active alerts.
    ///
    /// Returns `None` if there are no alerts.
    pub fn peak_priority(&self) -> Option<AlertPriority> {
        self.alerts
            .values()
            .filter(|r| matches!(r.status, AlertStatus::Pending | AlertStatus::Active))
            .map(|r| r.header.event.priority)
            .max()
    }

    /// Return `true` if any active or pending alert requires an immediate
    /// playout interrupt.
    pub fn requires_interrupt(&self) -> bool {
        self.alerts
            .values()
            .filter(|r| matches!(r.status, AlertStatus::Pending | AlertStatus::Active))
            .any(|r| r.header.event.priority.requires_interrupt())
    }

    /// Return a reference to the full interrupt event log.
    pub fn interrupt_log(&self) -> &[InterruptEvent] {
        &self.interrupt_log
    }

    /// Current frame position.
    pub fn current_frame(&self) -> u64 {
        self.current_frame
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── AlertPriority ─────────────────────────────────────────────────────────

    #[test]
    fn test_priority_ordering() {
        assert!(AlertPriority::Presidential > AlertPriority::Emergency);
        assert!(AlertPriority::Emergency > AlertPriority::Warning);
        assert!(AlertPriority::Warning > AlertPriority::Watch);
        assert!(AlertPriority::Watch > AlertPriority::Advisory);
        assert!(AlertPriority::Advisory > AlertPriority::Test);
    }

    #[test]
    fn test_priority_requires_interrupt() {
        assert!(AlertPriority::Warning.requires_interrupt());
        assert!(AlertPriority::Emergency.requires_interrupt());
        assert!(AlertPriority::Presidential.requires_interrupt());
        assert!(!AlertPriority::Watch.requires_interrupt());
        assert!(!AlertPriority::Test.requires_interrupt());
    }

    #[test]
    fn test_priority_label() {
        assert_eq!(AlertPriority::Emergency.label(), "EMERGENCY");
        assert_eq!(AlertPriority::Test.label(), "TEST");
    }

    // ── EasEventCode ─────────────────────────────────────────────────────────

    #[test]
    fn test_lookup_known_tornado_warning() {
        let code = EasEventCode::lookup("TOR").expect("TOR should be known");
        assert_eq!(code.priority, AlertPriority::Warning);
    }

    #[test]
    fn test_lookup_ean_is_presidential() {
        let code = EasEventCode::lookup("EAN").expect("EAN should be known");
        assert_eq!(code.priority, AlertPriority::Presidential);
    }

    #[test]
    fn test_lookup_unknown_returns_none() {
        assert!(EasEventCode::lookup("XXX").is_none());
    }

    #[test]
    fn test_unknown_event_code_advisory_priority() {
        let code = EasEventCode::unknown("ZZZ");
        assert_eq!(code.priority, AlertPriority::Advisory);
    }

    #[test]
    fn test_lookup_rwt_is_test() {
        let code = EasEventCode::lookup("RWT").expect("RWT should be known");
        assert_eq!(code.priority, AlertPriority::Test);
    }

    // ── SAME header parser ────────────────────────────────────────────────────

    /// Build a minimal valid SAME header string.
    fn sample_same_header() -> &'static str {
        // ZCZC-WXR-TOR-012345+0030-0151530-KABC/TV--
        "ZCZC-WXR-TOR-012345+0030-0151530-KABC/TV-"
    }

    #[test]
    fn test_parse_valid_same_header() {
        let header = SameHeader::parse(sample_same_header()).expect("should parse");
        assert_eq!(header.originator, "WXR");
        assert_eq!(header.event.code, "TOR");
        assert_eq!(header.event.priority, AlertPriority::Warning);
        assert_eq!(header.purge_hours, 0);
        assert_eq!(header.purge_minutes, 30);
        assert!(!header.locations.is_empty());
    }

    #[test]
    fn test_parse_ean_header() {
        // Presidential alert header
        let s = "ZCZC-PEP-EAN-000000+9999-0010000-WHITEHOUSE--";
        let header = SameHeader::parse(s).expect("should parse EAN");
        assert_eq!(header.event.priority, AlertPriority::Presidential);
        assert!(!header.is_test());
    }

    #[test]
    fn test_parse_test_header() {
        let s = "ZCZC-EAS-RWT-000000+0015-0011200-TESTSTAT-";
        let header = SameHeader::parse(s).expect("should parse test");
        assert!(header.is_test());
    }

    #[test]
    fn test_purge_duration_secs() {
        let s = "ZCZC-WXR-TOR-012345+0130-0151530-KABC/TV-";
        let header = SameHeader::parse(s).expect("should parse");
        // 1 hour 30 minutes = 5400 seconds
        assert_eq!(header.purge_duration_secs(), 5400);
    }

    #[test]
    fn test_parse_too_few_fields_error() {
        let result = SameHeader::parse("ZCZC-WXR-TOR");
        assert!(matches!(result, Err(ParseError::TooFewFields { .. })));
    }

    // ── AlertScheduler ────────────────────────────────────────────────────────

    fn make_tor_header() -> SameHeader {
        SameHeader::parse("ZCZC-WXR-TOR-012345+0030-0151530-KABC/TV-")
            .expect("valid header for test")
    }

    fn make_rwt_header() -> SameHeader {
        SameHeader::parse("ZCZC-EAS-RWT-000000+0015-0011200-TESTSTAT-")
            .expect("valid header for test")
    }

    #[test]
    fn test_admit_and_activate_warning() {
        let config = AlertSchedulerConfig::default();
        let mut sched = AlertScheduler::new(config);
        let id = sched.admit(make_tor_header(), 0).expect("admit TOR");
        let events = sched.advance_to_frame(0);
        assert!(events.iter().any(|e| e.alert_id == id && e.kind == InterruptEventKind::Begin));
    }

    #[test]
    fn test_requires_interrupt_with_warning() {
        let config = AlertSchedulerConfig::default();
        let mut sched = AlertScheduler::new(config);
        sched.admit(make_tor_header(), 0).expect("admit TOR");
        sched.advance_to_frame(0);
        assert!(sched.requires_interrupt());
    }

    #[test]
    fn test_complete_alert_transitions_to_pending_ack() {
        let config = AlertSchedulerConfig::default();
        let mut sched = AlertScheduler::new(config);
        let id = sched.admit(make_tor_header(), 0).expect("admit");
        sched.advance_to_frame(0);
        sched.complete_alert(id).expect("complete");
        assert_eq!(sched.record(id).expect("record").status, AlertStatus::PendingAck);
    }

    #[test]
    fn test_acknowledge_transitions_to_acknowledged() {
        let config = AlertSchedulerConfig::default();
        let mut sched = AlertScheduler::new(config);
        let id = sched.admit(make_tor_header(), 0).expect("admit");
        sched.advance_to_frame(0);
        sched.complete_alert(id).expect("complete");
        sched.acknowledge(id, "RELAY-1").expect("ack");
        assert_eq!(sched.record(id).expect("record").status, AlertStatus::Acknowledged);
        assert_eq!(sched.record(id).expect("record").acks.len(), 1);
    }

    #[test]
    fn test_cancel_pending_alert() {
        let config = AlertSchedulerConfig::default();
        let mut sched = AlertScheduler::new(config);
        let id = sched.admit(make_tor_header(), 1000).expect("admit future alert");
        sched.cancel(id).expect("cancel");
        assert_eq!(sched.record(id).expect("record").status, AlertStatus::Cancelled);
        assert!(sched.pending_ids().is_empty());
    }

    #[test]
    fn test_below_minimum_priority_rejected() {
        let config = AlertSchedulerConfig {
            min_priority: AlertPriority::Warning,
            admit_tests: false,
            ..Default::default()
        };
        let mut sched = AlertScheduler::new(config);
        // Advisory is below Warning
        let adv_header = SameHeader {
            originator: "CIV".to_string(),
            event: EasEventCode { code: "ADR".to_string(), description: "Admin".to_string(), priority: AlertPriority::Statement },
            locations: vec!["012345".to_string()],
            purge_hours: 0,
            purge_minutes: 30,
            issue_julian_day: 15,
            issue_hour: 15,
            issue_minute: 30,
            station_id: "TEST".to_string(),
        };
        let result = sched.admit(adv_header, 0);
        assert!(matches!(result, Err(AlertSchedulerError::BelowMinimumPriority { .. })));
    }

    #[test]
    fn test_admit_test_even_if_below_min_when_admit_tests_true() {
        let config = AlertSchedulerConfig {
            min_priority: AlertPriority::Warning,
            admit_tests: true,
            ..Default::default()
        };
        let mut sched = AlertScheduler::new(config);
        // RWT is a test — should be admitted
        let result = sched.admit(make_rwt_header(), 0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_peak_priority_with_multiple_alerts() {
        let config = AlertSchedulerConfig::default();
        let mut sched = AlertScheduler::new(config);
        // Admit a severe thunderstorm watch (Watch priority) and a tornado warning (Warning priority)
        let watch = SameHeader::parse("ZCZC-WXR-SVA-012345+0030-0151530-KABC/TV-").expect("SVA");
        sched.admit(watch, 0).expect("admit watch");
        sched.admit(make_tor_header(), 0).expect("admit TOR");
        // Peak priority should be Warning (the highest)
        assert_eq!(sched.peak_priority(), Some(AlertPriority::Warning));
    }

    #[test]
    fn test_interrupt_log_accumulates() {
        let config = AlertSchedulerConfig::default();
        let mut sched = AlertScheduler::new(config);
        let id = sched.admit(make_tor_header(), 0).expect("admit");
        sched.advance_to_frame(0);
        sched.complete_alert(id).expect("complete");
        // Should have Begin and End events
        let log = sched.interrupt_log();
        assert!(log.iter().any(|e| e.kind == InterruptEventKind::Begin));
        assert!(log.iter().any(|e| e.kind == InterruptEventKind::End));
    }

    #[test]
    fn test_max_concurrent_respected() {
        let config = AlertSchedulerConfig {
            max_concurrent: 1,
            ..Default::default()
        };
        let mut sched = AlertScheduler::new(config);
        let h1 = SameHeader::parse("ZCZC-WXR-TOR-012345+0030-0151530-KABC/TV-").expect("h1");
        let h2 = SameHeader::parse("ZCZC-WXR-SVR-067890+0030-0151530-KABC/TV-").expect("h2");
        let id1 = sched.admit(h1, 0).expect("admit 1");
        let id2 = sched.admit(h2, 0).expect("admit 2");
        sched.advance_to_frame(0);
        // Only one should be active
        assert_eq!(sched.active_alert_ids().len(), 1);
        // The other should still be pending
        let id1_status = sched.record(id1).expect("record 1").status;
        let id2_status = sched.record(id2).expect("record 2").status;
        let active_count = [id1_status, id2_status]
            .iter()
            .filter(|&&s| s == AlertStatus::Active)
            .count();
        let pending_count = [id1_status, id2_status]
            .iter()
            .filter(|&&s| s == AlertStatus::Pending)
            .count();
        assert_eq!(active_count, 1);
        assert_eq!(pending_count, 1);
    }
}
