#![allow(dead_code)]
//! # Playout Schedule
//!
//! Time-grid schedule management for broadcast playout. Tracks scheduled
//! programme slots across a 24-hour day, supports conflict detection, gap
//! finding, and frame-accurate start-time queries.

use std::collections::BTreeMap;
use std::time::Duration;

/// A single scheduled slot on the playout timeline.
#[derive(Debug, Clone, PartialEq)]
pub struct ScheduleSlot {
    /// Unique identifier for this slot.
    pub id: u64,
    /// Offset from midnight (seconds).
    pub start_secs: u64,
    /// Duration of the slot.
    pub duration: Duration,
    /// Human-readable title.
    pub title: String,
    /// Whether the slot is locked against edits.
    pub locked: bool,
}

impl ScheduleSlot {
    /// Create a new schedule slot.
    pub fn new(id: u64, start_secs: u64, duration: Duration, title: impl Into<String>) -> Self {
        Self {
            id,
            start_secs,
            duration,
            title: title.into(),
            locked: false,
        }
    }

    /// Return the second at which this slot ends.
    #[allow(clippy::cast_possible_truncation)]
    pub fn end_secs(&self) -> u64 {
        self.start_secs + self.duration.as_secs()
    }

    /// Return `true` if this slot overlaps with `other`.
    pub fn overlaps(&self, other: &ScheduleSlot) -> bool {
        self.start_secs < other.end_secs() && other.start_secs < self.end_secs()
    }

    /// Lock this slot against modification.
    pub fn lock(&mut self) {
        self.locked = true;
    }

    /// Unlock this slot.
    pub fn unlock(&mut self) {
        self.locked = false;
    }
}

/// Priority level for schedule slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum SlotPriority {
    /// Filler / default content.
    Low,
    /// Regular programming.
    #[default]
    Normal,
    /// Breaking news or live events.
    High,
    /// Emergency override (cannot be displaced).
    Emergency,
}

/// Result type for playout schedule operations.
pub type ScheduleResult<T> = Result<T, ScheduleError>;

/// Errors that can occur during schedule operations.
#[derive(Debug, Clone, PartialEq)]
pub enum ScheduleError {
    /// The requested slot ID was not found.
    NotFound(u64),
    /// The slot conflicts with an existing locked slot.
    Conflict { existing_id: u64 },
    /// The slot duration is zero or invalid.
    InvalidDuration,
    /// The slot start time is outside the 24-hour day (≥ 86 400 s).
    InvalidStartTime,
    /// A locked slot cannot be modified.
    Locked(u64),
}

impl std::fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScheduleError::NotFound(id) => write!(f, "Slot {id} not found"),
            ScheduleError::Conflict { existing_id } => {
                write!(f, "Conflicts with locked slot {existing_id}")
            }
            ScheduleError::InvalidDuration => write!(f, "Slot duration must be non-zero"),
            ScheduleError::InvalidStartTime => write!(f, "Start time must be < 86400 seconds"),
            ScheduleError::Locked(id) => write!(f, "Slot {id} is locked"),
        }
    }
}

/// 24-hour playout schedule grid.
///
/// Slots are stored sorted by start time. Overlap is allowed unless the
/// conflicting slot is locked, in which case an error is returned.
pub struct PlayoutSchedule {
    slots: BTreeMap<u64, ScheduleSlot>, // keyed by slot id
    next_id: u64,
}

impl PlayoutSchedule {
    /// Create an empty schedule.
    pub fn new() -> Self {
        Self {
            slots: BTreeMap::new(),
            next_id: 1,
        }
    }

    /// Add a slot to the schedule. Returns the assigned slot ID.
    ///
    /// Fails if the new slot overlaps a *locked* existing slot.
    pub fn add_slot(
        &mut self,
        start_secs: u64,
        duration: Duration,
        title: impl Into<String>,
    ) -> ScheduleResult<u64> {
        if duration.is_zero() {
            return Err(ScheduleError::InvalidDuration);
        }
        if start_secs >= 86_400 {
            return Err(ScheduleError::InvalidStartTime);
        }
        let id = self.next_id;
        self.next_id += 1;
        let candidate = ScheduleSlot::new(id, start_secs, duration, title);

        // Check for conflicts with locked slots
        for slot in self.slots.values() {
            if slot.locked && candidate.overlaps(slot) {
                return Err(ScheduleError::Conflict {
                    existing_id: slot.id,
                });
            }
        }

        self.slots.insert(id, candidate);
        Ok(id)
    }

    /// Remove a slot by ID. Returns an error if the slot is locked.
    pub fn remove_slot(&mut self, id: u64) -> ScheduleResult<ScheduleSlot> {
        let slot = self.slots.get(&id).ok_or(ScheduleError::NotFound(id))?;
        if slot.locked {
            return Err(ScheduleError::Locked(id));
        }
        self.slots.remove(&id).ok_or(ScheduleError::NotFound(id))
    }

    /// Lock a slot, preventing removal or time edits.
    pub fn lock_slot(&mut self, id: u64) -> ScheduleResult<()> {
        self.slots
            .get_mut(&id)
            .ok_or(ScheduleError::NotFound(id))
            .map(ScheduleSlot::lock)
    }

    /// Unlock a slot.
    pub fn unlock_slot(&mut self, id: u64) -> ScheduleResult<()> {
        self.slots
            .get_mut(&id)
            .ok_or(ScheduleError::NotFound(id))
            .map(ScheduleSlot::unlock)
    }

    /// Return the slot that should be on-air at `playhead_secs`.
    pub fn current_slot(&self, playhead_secs: u64) -> Option<&ScheduleSlot> {
        self.slots
            .values()
            .find(|s| s.start_secs <= playhead_secs && playhead_secs < s.end_secs())
    }

    /// Return the next slot that starts after `playhead_secs`.
    pub fn next_slot(&self, playhead_secs: u64) -> Option<&ScheduleSlot> {
        self.slots
            .values()
            .filter(|s| s.start_secs > playhead_secs)
            .min_by_key(|s| s.start_secs)
    }

    /// Return all slots sorted by start time.
    pub fn sorted_slots(&self) -> Vec<&ScheduleSlot> {
        let mut v: Vec<&ScheduleSlot> = self.slots.values().collect();
        v.sort_by_key(|s| s.start_secs);
        v
    }

    /// Find gaps (unscheduled intervals) within `[from_secs, to_secs)`.
    ///
    /// Returns a list of `(gap_start, gap_end)` pairs.
    pub fn gaps(&self, from_secs: u64, to_secs: u64) -> Vec<(u64, u64)> {
        let mut result = Vec::new();
        let mut cursor = from_secs;
        for slot in self.sorted_slots() {
            if slot.start_secs >= to_secs {
                break;
            }
            if slot.start_secs > cursor {
                result.push((cursor, slot.start_secs));
            }
            if slot.end_secs() > cursor {
                cursor = slot.end_secs();
            }
        }
        if cursor < to_secs {
            result.push((cursor, to_secs));
        }
        result
    }

    /// Total number of slots.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Return `true` if no slots are scheduled.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Return a slot by ID.
    pub fn get(&self, id: u64) -> Option<&ScheduleSlot> {
        self.slots.get(&id)
    }
}

impl Default for PlayoutSchedule {
    fn default() -> Self {
        Self::new()
    }
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn dur(secs: u64) -> Duration {
        Duration::from_secs(secs)
    }

    #[test]
    fn test_slot_end_secs() {
        let s = ScheduleSlot::new(1, 3600, dur(1800), "Morning Show");
        assert_eq!(s.end_secs(), 5400);
    }

    #[test]
    fn test_slot_overlap_true() {
        let a = ScheduleSlot::new(1, 0, dur(3600), "A");
        let b = ScheduleSlot::new(2, 1800, dur(3600), "B");
        assert!(a.overlaps(&b));
    }

    #[test]
    fn test_slot_overlap_false() {
        let a = ScheduleSlot::new(1, 0, dur(3600), "A");
        let b = ScheduleSlot::new(2, 3600, dur(3600), "B");
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_add_slot_returns_id() {
        let mut sched = PlayoutSchedule::new();
        let id = sched
            .add_slot(0, dur(3600), "News")
            .expect("should succeed in test");
        assert_eq!(id, 1);
    }

    #[test]
    fn test_add_multiple_slots() {
        let mut sched = PlayoutSchedule::new();
        let id1 = sched
            .add_slot(0, dur(1800), "A")
            .expect("should succeed in test");
        let id2 = sched
            .add_slot(1800, dur(1800), "B")
            .expect("should succeed in test");
        assert_ne!(id1, id2);
        assert_eq!(sched.len(), 2);
    }

    #[test]
    fn test_zero_duration_rejected() {
        let mut sched = PlayoutSchedule::new();
        let err = sched.add_slot(0, Duration::ZERO, "Bad").unwrap_err();
        assert_eq!(err, ScheduleError::InvalidDuration);
    }

    #[test]
    fn test_invalid_start_time() {
        let mut sched = PlayoutSchedule::new();
        let err = sched.add_slot(86_400, dur(60), "Midnight+1").unwrap_err();
        assert_eq!(err, ScheduleError::InvalidStartTime);
    }

    #[test]
    fn test_conflict_with_locked_slot() {
        let mut sched = PlayoutSchedule::new();
        let id = sched
            .add_slot(0, dur(3600), "Locked News")
            .expect("should succeed in test");
        sched.lock_slot(id).expect("should succeed in test");
        let err = sched.add_slot(1800, dur(3600), "Overlap").unwrap_err();
        assert!(matches!(err, ScheduleError::Conflict { .. }));
    }

    #[test]
    fn test_remove_unlocked_slot() {
        let mut sched = PlayoutSchedule::new();
        let id = sched
            .add_slot(0, dur(3600), "Remove Me")
            .expect("should succeed in test");
        let removed = sched.remove_slot(id).expect("should succeed in test");
        assert_eq!(removed.id, id);
        assert!(sched.is_empty());
    }

    #[test]
    fn test_remove_locked_slot_fails() {
        let mut sched = PlayoutSchedule::new();
        let id = sched
            .add_slot(0, dur(3600), "Locked")
            .expect("should succeed in test");
        sched.lock_slot(id).expect("should succeed in test");
        assert!(matches!(
            sched.remove_slot(id).unwrap_err(),
            ScheduleError::Locked(_)
        ));
    }

    #[test]
    fn test_current_slot_found() {
        let mut sched = PlayoutSchedule::new();
        sched
            .add_slot(3600, dur(1800), "Slot A")
            .expect("should succeed in test");
        let slot = sched.current_slot(4000).expect("should succeed in test");
        assert_eq!(slot.title, "Slot A");
    }

    #[test]
    fn test_current_slot_none() {
        let mut sched = PlayoutSchedule::new();
        sched
            .add_slot(3600, dur(1800), "Slot A")
            .expect("should succeed in test");
        assert!(sched.current_slot(0).is_none());
    }

    #[test]
    fn test_next_slot() {
        let mut sched = PlayoutSchedule::new();
        sched
            .add_slot(0, dur(1800), "First")
            .expect("should succeed in test");
        sched
            .add_slot(1800, dur(1800), "Second")
            .expect("should succeed in test");
        let next = sched.next_slot(500).expect("should succeed in test");
        assert_eq!(next.title, "Second");
    }

    #[test]
    fn test_gaps_detection() {
        let mut sched = PlayoutSchedule::new();
        sched
            .add_slot(1000, dur(500), "A")
            .expect("should succeed in test");
        sched
            .add_slot(2000, dur(500), "B")
            .expect("should succeed in test");
        let gaps = sched.gaps(0, 3000);
        assert_eq!(gaps, vec![(0, 1000), (1500, 2000), (2500, 3000)]);
    }

    #[test]
    fn test_sorted_slots_order() {
        let mut sched = PlayoutSchedule::new();
        sched
            .add_slot(7200, dur(1800), "Late")
            .expect("should succeed in test");
        sched
            .add_slot(0, dur(1800), "Early")
            .expect("should succeed in test");
        let sorted = sched.sorted_slots();
        assert_eq!(sorted[0].title, "Early");
        assert_eq!(sorted[1].title, "Late");
    }

    #[test]
    fn test_lock_unlock_cycle() {
        let mut sched = PlayoutSchedule::new();
        let id = sched
            .add_slot(0, dur(3600), "Toggle")
            .expect("should succeed in test");
        sched.lock_slot(id).expect("should succeed in test");
        assert!(sched.get(id).expect("should succeed in test").locked);
        sched.unlock_slot(id).expect("should succeed in test");
        assert!(!sched.get(id).expect("should succeed in test").locked);
    }

    #[test]
    fn test_slot_priority_ordering() {
        assert!(SlotPriority::Low < SlotPriority::Normal);
        assert!(SlotPriority::Normal < SlotPriority::High);
        assert!(SlotPriority::High < SlotPriority::Emergency);
    }
}
