//! Branding inserter: frame-accurate scheduling and management of on-screen
//! branding elements.
//!
//! This module provides the machinery needed to schedule, enable, and disable
//! branding overlays (channel bugs, watermarks, end-boards, promotional
//! slates, etc.) during a live playout session.
//!
//! # Key types
//!
//! - [`BrandingSlot`] — a single scheduled occurrence of a branding element.
//! - [`BrandingInserter`] — the central scheduler that tracks which elements
//!   are active at any given timeline position and fires cue events.
//! - [`InsertionPolicy`] — controls how conflicting overlays of the same type
//!   are handled.
//! - [`BrandingEvent`] — the output cue event produced when an element should
//!   be activated or deactivated.
//!
//! # Usage
//!
//! ```rust
//! use oximedia_playout::branding_inserter::{
//!     BrandingInserter, BrandingSlot, BrandingZone, InsertionPolicy,
//! };
//!
//! let mut inserter = BrandingInserter::new(InsertionPolicy::ReplaceOnConflict);
//!
//! // Schedule the channel bug for the entire programme.
//! inserter.schedule(BrandingSlot {
//!     id: 1,
//!     asset_id: 10,
//!     zone: BrandingZone::BugTopRight,
//!     start_frame: 0,
//!     end_frame: None,   // persistent
//!     priority: 5,
//!     label: "Channel Bug".into(),
//! }).expect("schedule bug");
//!
//! let events = inserter.advance_to_frame(0);
//! assert_eq!(events.len(), 1);
//! ```

#![allow(dead_code)]

use std::collections::HashMap;

// ── Position zone ──────────────────────────────────────────────────────────────

/// Screen zone where a branding element is rendered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BrandingZone {
    /// Top-left corner (e.g., channel logo / bug)
    BugTopLeft,
    /// Top-right corner
    BugTopRight,
    /// Bottom-left corner
    BugBottomLeft,
    /// Bottom-right corner (e.g., ratings badge)
    BugBottomRight,
    /// Centre of frame (e.g., promotional slate)
    CentreFill,
    /// Lower-third banner area
    LowerThird,
    /// Full-frame overlay (e.g., end-board, DOG)
    FullFrame,
}

impl BrandingZone {
    /// Returns `true` if this zone can be layered over programme content
    /// without blocking it entirely.
    pub fn is_bug_zone(&self) -> bool {
        matches!(
            self,
            Self::BugTopLeft
                | Self::BugTopRight
                | Self::BugBottomLeft
                | Self::BugBottomRight
        )
    }

    /// Returns `true` if this zone occupies the full raster.
    pub fn is_full_frame(&self) -> bool {
        matches!(self, Self::FullFrame | Self::CentreFill)
    }
}

// ── Insertion policy ───────────────────────────────────────────────────────────

/// Determines how the inserter resolves scheduling conflicts in the same zone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertionPolicy {
    /// A new overlay replaces any existing one in the same zone.
    ReplaceOnConflict,
    /// The highest-priority slot wins; ties go to the existing slot.
    PriorityWins,
    /// Multiple overlays are allowed simultaneously (compositor handles layering).
    AllowOverlap,
}

// ── Branding slot ─────────────────────────────────────────────────────────────

/// A single scheduled occurrence of a branding element on the timeline.
#[derive(Debug, Clone, PartialEq)]
pub struct BrandingSlot {
    /// Unique slot identifier.
    pub id: u32,
    /// ID of the [`crate::branding::BrandingAsset`] to display.
    pub asset_id: u32,
    /// Screen zone for this overlay.
    pub zone: BrandingZone,
    /// Frame number at which the overlay should first appear.
    pub start_frame: u64,
    /// Frame number at which the overlay should be removed.
    /// `None` means the overlay persists indefinitely.
    pub end_frame: Option<u64>,
    /// Scheduling priority (higher = more important).
    pub priority: u8,
    /// Human-readable label for logging / debugging.
    pub label: String,
}

impl BrandingSlot {
    /// Returns `true` if this slot is active at the given frame number.
    pub fn is_active_at(&self, frame: u64) -> bool {
        if frame < self.start_frame {
            return false;
        }
        match self.end_frame {
            Some(end) => frame < end,
            None => true,
        }
    }

    /// Duration in frames, or `None` if the slot is persistent.
    pub fn duration_frames(&self) -> Option<u64> {
        self.end_frame.map(|e| e.saturating_sub(self.start_frame))
    }
}

// ── Branding events ────────────────────────────────────────────────────────────

/// The kind of event emitted by the inserter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrandingEventKind {
    /// A branding element should become visible.
    Activate,
    /// A branding element should be removed.
    Deactivate,
    /// A branding element that was active has been superseded and must be
    /// replaced with a higher-priority overlay.
    Superseded,
}

/// A cue event emitted when a slot transitions.
#[derive(Debug, Clone)]
pub struct BrandingEvent {
    /// The slot that triggered this event.
    pub slot_id: u32,
    /// Asset to show / hide.
    pub asset_id: u32,
    /// Zone affected.
    pub zone: BrandingZone,
    /// Kind of transition.
    pub kind: BrandingEventKind,
    /// Frame number at which the event fires.
    pub frame: u64,
}

// ── Errors ─────────────────────────────────────────────────────────────────────

/// Errors returned by the branding inserter.
#[derive(Debug, Clone, PartialEq)]
pub enum InserterError {
    /// A slot with the same ID has already been scheduled.
    DuplicateSlotId(u32),
    /// The specified slot ID was not found.
    SlotNotFound(u32),
    /// End frame is before start frame.
    InvalidFrameRange { start: u64, end: u64 },
    /// Zone conflict: policy prevents the slot from being scheduled.
    ZoneConflict { zone: BrandingZone, existing_slot_id: u32 },
}

impl std::fmt::Display for InserterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateSlotId(id) => write!(f, "slot ID {id} already scheduled"),
            Self::SlotNotFound(id) => write!(f, "slot ID {id} not found"),
            Self::InvalidFrameRange { start, end } => {
                write!(f, "end frame {end} is before start frame {start}")
            }
            Self::ZoneConflict { zone, existing_slot_id } => {
                write!(f, "zone {zone:?} already occupied by slot {existing_slot_id}")
            }
        }
    }
}

impl std::error::Error for InserterError {}

// ── Branding inserter ─────────────────────────────────────────────────────────

/// Central branding scheduler and cue event generator.
///
/// Maintains a registry of scheduled [`BrandingSlot`]s and tracks the current
/// frame position to emit [`BrandingEvent`]s at the correct moment.
#[derive(Debug, Clone)]
pub struct BrandingInserter {
    /// All scheduled slots, keyed by slot ID.
    slots: HashMap<u32, BrandingSlot>,
    /// Slots currently active (displayed), keyed by zone.
    /// Maps zone → set of active slot IDs (for AllowOverlap policy).
    active_by_zone: HashMap<BrandingZone, Vec<u32>>,
    /// Current frame cursor.
    current_frame: u64,
    /// Conflict-resolution policy.
    policy: InsertionPolicy,
    /// Event log (all events ever emitted in this session).
    event_log: Vec<BrandingEvent>,
}

impl BrandingInserter {
    /// Create a new inserter with the given conflict policy.
    pub fn new(policy: InsertionPolicy) -> Self {
        Self {
            slots: HashMap::new(),
            active_by_zone: HashMap::new(),
            current_frame: 0,
            policy,
            event_log: Vec::new(),
        }
    }

    // ── Scheduling ────────────────────────────────────────────────────────────

    /// Schedule a [`BrandingSlot`] for future playback.
    ///
    /// Returns an error if the ID is a duplicate, the frame range is invalid,
    /// or the zone is already occupied and the policy prevents scheduling.
    pub fn schedule(&mut self, slot: BrandingSlot) -> Result<(), InserterError> {
        // Validate frame range
        if let Some(end) = slot.end_frame {
            if end <= slot.start_frame {
                return Err(InserterError::InvalidFrameRange {
                    start: slot.start_frame,
                    end,
                });
            }
        }

        // Duplicate ID check
        if self.slots.contains_key(&slot.id) {
            return Err(InserterError::DuplicateSlotId(slot.id));
        }

        self.slots.insert(slot.id, slot);
        Ok(())
    }

    /// Remove a scheduled slot by ID.
    ///
    /// If the slot is currently active, a [`BrandingEventKind::Deactivate`]
    /// event is emitted.
    pub fn cancel(&mut self, slot_id: u32) -> Result<Vec<BrandingEvent>, InserterError> {
        let slot = self.slots.remove(&slot_id).ok_or(InserterError::SlotNotFound(slot_id))?;
        let mut events = Vec::new();

        // Check if active in any zone
        if let Some(active_ids) = self.active_by_zone.get_mut(&slot.zone) {
            if let Some(pos) = active_ids.iter().position(|id| *id == slot_id) {
                active_ids.remove(pos);
                if active_ids.is_empty() {
                    self.active_by_zone.remove(&slot.zone);
                }
                let ev = BrandingEvent {
                    slot_id,
                    asset_id: slot.asset_id,
                    zone: slot.zone,
                    kind: BrandingEventKind::Deactivate,
                    frame: self.current_frame,
                };
                self.event_log.push(ev.clone());
                events.push(ev);
            }
        }

        Ok(events)
    }

    // ── Frame advance ─────────────────────────────────────────────────────────

    /// Advance the internal cursor to `target_frame` and return all cue events
    /// that fire between the previous cursor position and `target_frame`
    /// (inclusive).
    ///
    /// Events are sorted by slot priority (descending) within each frame.
    pub fn advance_to_frame(&mut self, target_frame: u64) -> Vec<BrandingEvent> {
        let mut events = Vec::new();

        // Collect slot IDs to examine (avoid borrow conflicts)
        let slot_ids: Vec<u32> = self.slots.keys().copied().collect();

        for slot_id in &slot_ids {
            let slot = match self.slots.get(slot_id) {
                Some(s) => s.clone(),
                None => continue,
            };

            // Was this slot active before this advance?
            let was_active = self
                .active_by_zone
                .get(&slot.zone)
                .map(|ids| ids.contains(slot_id))
                .unwrap_or(false);

            // Is the slot active at the new target frame?
            let is_active_now = slot.is_active_at(target_frame);

            match (was_active, is_active_now) {
                (false, true) => {
                    // Need to activate — check policy
                    let conflict = self.find_active_in_zone(&slot.zone, *slot_id);

                    match self.policy {
                        InsertionPolicy::AllowOverlap => {
                            self.mark_active(*slot_id, slot.zone);
                            let ev = BrandingEvent {
                                slot_id: *slot_id,
                                asset_id: slot.asset_id,
                                zone: slot.zone,
                                kind: BrandingEventKind::Activate,
                                frame: target_frame,
                            };
                            self.event_log.push(ev.clone());
                            events.push(ev);
                        }
                        InsertionPolicy::ReplaceOnConflict => {
                            if let Some(existing_id) = conflict {
                                // Deactivate the existing one
                                if let Some(existing) = self.slots.get(&existing_id).cloned() {
                                    self.unmark_active(existing_id, existing.zone);
                                    let ev = BrandingEvent {
                                        slot_id: existing_id,
                                        asset_id: existing.asset_id,
                                        zone: existing.zone,
                                        kind: BrandingEventKind::Superseded,
                                        frame: target_frame,
                                    };
                                    self.event_log.push(ev.clone());
                                    events.push(ev);
                                }
                            }
                            self.mark_active(*slot_id, slot.zone);
                            let ev = BrandingEvent {
                                slot_id: *slot_id,
                                asset_id: slot.asset_id,
                                zone: slot.zone,
                                kind: BrandingEventKind::Activate,
                                frame: target_frame,
                            };
                            self.event_log.push(ev.clone());
                            events.push(ev);
                        }
                        InsertionPolicy::PriorityWins => {
                            let existing_priority = conflict.and_then(|eid| {
                                self.slots.get(&eid).map(|s| s.priority)
                            });
                            let should_activate = match existing_priority {
                                None => true,
                                Some(ep) => slot.priority > ep,
                            };
                            if should_activate {
                                if let Some(existing_id) = conflict {
                                    if let Some(existing) = self.slots.get(&existing_id).cloned() {
                                        self.unmark_active(existing_id, existing.zone);
                                        let ev = BrandingEvent {
                                            slot_id: existing_id,
                                            asset_id: existing.asset_id,
                                            zone: existing.zone,
                                            kind: BrandingEventKind::Superseded,
                                            frame: target_frame,
                                        };
                                        self.event_log.push(ev.clone());
                                        events.push(ev);
                                    }
                                }
                                self.mark_active(*slot_id, slot.zone);
                                let ev = BrandingEvent {
                                    slot_id: *slot_id,
                                    asset_id: slot.asset_id,
                                    zone: slot.zone,
                                    kind: BrandingEventKind::Activate,
                                    frame: target_frame,
                                };
                                self.event_log.push(ev.clone());
                                events.push(ev);
                            }
                        }
                    }
                }
                (true, false) => {
                    // Slot has expired — deactivate
                    self.unmark_active(*slot_id, slot.zone);
                    let ev = BrandingEvent {
                        slot_id: *slot_id,
                        asset_id: slot.asset_id,
                        zone: slot.zone,
                        kind: BrandingEventKind::Deactivate,
                        frame: target_frame,
                    };
                    self.event_log.push(ev.clone());
                    events.push(ev);
                }
                _ => {}
            }
        }

        // Sort by priority desc, then slot_id for deterministic output
        events.sort_by(|a, b| {
            let pa = self.slots.get(&a.slot_id).map(|s| s.priority).unwrap_or(0);
            let pb = self.slots.get(&b.slot_id).map(|s| s.priority).unwrap_or(0);
            pb.cmp(&pa).then(a.slot_id.cmp(&b.slot_id))
        });

        self.current_frame = target_frame;
        events
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// Return all slot IDs that are active at the current frame.
    pub fn active_slot_ids(&self) -> Vec<u32> {
        self.active_by_zone
            .values()
            .flat_map(|ids| ids.iter().copied())
            .collect()
    }

    /// Return the current frame position.
    pub fn current_frame(&self) -> u64 {
        self.current_frame
    }

    /// Return all events ever emitted in this session.
    pub fn event_log(&self) -> &[BrandingEvent] {
        &self.event_log
    }

    /// Return the number of slots currently scheduled (active or pending).
    pub fn scheduled_count(&self) -> usize {
        self.slots.len()
    }

    /// Return all slots scheduled in the given zone at any time.
    pub fn slots_in_zone(&self, zone: BrandingZone) -> Vec<&BrandingSlot> {
        self.slots.values().filter(|s| s.zone == zone).collect()
    }

    /// Return the slot with the given ID, if it exists.
    pub fn slot(&self, id: u32) -> Option<&BrandingSlot> {
        self.slots.get(&id)
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn mark_active(&mut self, slot_id: u32, zone: BrandingZone) {
        self.active_by_zone
            .entry(zone)
            .or_default()
            .push(slot_id);
    }

    fn unmark_active(&mut self, slot_id: u32, zone: BrandingZone) {
        if let Some(ids) = self.active_by_zone.get_mut(&zone) {
            ids.retain(|id| *id != slot_id);
            if ids.is_empty() {
                self.active_by_zone.remove(&zone);
            }
        }
    }

    /// Find the first active slot in `zone` that isn't `exclude_id`.
    fn find_active_in_zone(&self, zone: &BrandingZone, exclude_id: u32) -> Option<u32> {
        self.active_by_zone
            .get(zone)?
            .iter()
            .copied()
            .find(|id| *id != exclude_id)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn bug_slot(id: u32, start: u64, end: Option<u64>, priority: u8) -> BrandingSlot {
        BrandingSlot {
            id,
            asset_id: id * 10,
            zone: BrandingZone::BugTopRight,
            start_frame: start,
            end_frame: end,
            priority,
            label: format!("Slot {id}"),
        }
    }

    fn lower_third_slot(id: u32, start: u64, end: Option<u64>, priority: u8) -> BrandingSlot {
        BrandingSlot {
            id,
            asset_id: id * 10,
            zone: BrandingZone::LowerThird,
            start_frame: start,
            end_frame: end,
            priority,
            label: format!("LT {id}"),
        }
    }

    // ── Slot helpers ──────────────────────────────────────────────────────────

    #[test]
    fn test_slot_is_active_at_within_range() {
        let slot = bug_slot(1, 100, Some(200), 5);
        assert!(slot.is_active_at(100));
        assert!(slot.is_active_at(150));
        assert!(!slot.is_active_at(200)); // exclusive end
    }

    #[test]
    fn test_slot_persistent_is_always_active() {
        let slot = bug_slot(1, 0, None, 5);
        assert!(slot.is_active_at(0));
        assert!(slot.is_active_at(999_999));
    }

    #[test]
    fn test_slot_duration_frames_calculated() {
        let slot = bug_slot(1, 50, Some(150), 5);
        assert_eq!(slot.duration_frames(), Some(100));
    }

    #[test]
    fn test_slot_duration_frames_persistent_is_none() {
        let slot = bug_slot(1, 0, None, 5);
        assert_eq!(slot.duration_frames(), None);
    }

    // ── BrandingZone helpers ──────────────────────────────────────────────────

    #[test]
    fn test_zone_is_bug_zone() {
        assert!(BrandingZone::BugTopLeft.is_bug_zone());
        assert!(BrandingZone::BugTopRight.is_bug_zone());
        assert!(!BrandingZone::LowerThird.is_bug_zone());
        assert!(!BrandingZone::FullFrame.is_bug_zone());
    }

    #[test]
    fn test_zone_is_full_frame() {
        assert!(BrandingZone::FullFrame.is_full_frame());
        assert!(BrandingZone::CentreFill.is_full_frame());
        assert!(!BrandingZone::BugTopLeft.is_full_frame());
    }

    // ── Schedule / cancel ─────────────────────────────────────────────────────

    #[test]
    fn test_schedule_and_count() {
        let mut ins = BrandingInserter::new(InsertionPolicy::AllowOverlap);
        ins.schedule(bug_slot(1, 0, None, 5)).expect("schedule");
        ins.schedule(bug_slot(2, 100, Some(200), 3)).expect("schedule");
        assert_eq!(ins.scheduled_count(), 2);
    }

    #[test]
    fn test_schedule_duplicate_id_error() {
        let mut ins = BrandingInserter::new(InsertionPolicy::AllowOverlap);
        ins.schedule(bug_slot(1, 0, None, 5)).expect("first schedule");
        let result = ins.schedule(bug_slot(1, 50, None, 3));
        assert!(matches!(result, Err(InserterError::DuplicateSlotId(1))));
    }

    #[test]
    fn test_schedule_invalid_frame_range_error() {
        let mut ins = BrandingInserter::new(InsertionPolicy::AllowOverlap);
        let mut slot = bug_slot(1, 200, Some(100), 5); // end < start
        slot.end_frame = Some(100);
        slot.start_frame = 200;
        let result = ins.schedule(slot);
        assert!(matches!(result, Err(InserterError::InvalidFrameRange { .. })));
    }

    #[test]
    fn test_cancel_active_slot_emits_deactivate() {
        let mut ins = BrandingInserter::new(InsertionPolicy::AllowOverlap);
        ins.schedule(bug_slot(1, 0, None, 5)).expect("schedule");
        ins.advance_to_frame(0); // activates slot 1
        let events = ins.cancel(1).expect("cancel");
        assert!(events.iter().any(|e| e.kind == BrandingEventKind::Deactivate));
    }

    #[test]
    fn test_cancel_unknown_slot_error() {
        let mut ins = BrandingInserter::new(InsertionPolicy::AllowOverlap);
        let result = ins.cancel(99);
        assert!(matches!(result, Err(InserterError::SlotNotFound(99))));
    }

    // ── Advance / events ──────────────────────────────────────────────────────

    #[test]
    fn test_advance_activates_persistent_slot_on_start() {
        let mut ins = BrandingInserter::new(InsertionPolicy::AllowOverlap);
        ins.schedule(bug_slot(1, 0, None, 5)).expect("schedule");
        let events = ins.advance_to_frame(0);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, BrandingEventKind::Activate);
        assert_eq!(events[0].slot_id, 1);
    }

    #[test]
    fn test_advance_deactivates_expired_slot() {
        let mut ins = BrandingInserter::new(InsertionPolicy::AllowOverlap);
        ins.schedule(bug_slot(1, 0, Some(50), 5)).expect("schedule");
        ins.advance_to_frame(0); // activate
        let events = ins.advance_to_frame(50); // should deactivate
        assert!(events.iter().any(|e| e.kind == BrandingEventKind::Deactivate && e.slot_id == 1));
    }

    #[test]
    fn test_replace_policy_supersedes_existing() {
        let mut ins = BrandingInserter::new(InsertionPolicy::ReplaceOnConflict);
        ins.schedule(bug_slot(1, 0, None, 3)).expect("schedule low-prio");
        ins.schedule(bug_slot(2, 50, None, 5)).expect("schedule high-prio");
        ins.advance_to_frame(0); // activate slot 1
        let events = ins.advance_to_frame(50);
        // Slot 1 should be superseded, slot 2 should be activated
        let superseded = events.iter().any(|e| e.slot_id == 1 && e.kind == BrandingEventKind::Superseded);
        let activated = events.iter().any(|e| e.slot_id == 2 && e.kind == BrandingEventKind::Activate);
        assert!(superseded, "slot 1 should be superseded");
        assert!(activated, "slot 2 should be activated");
    }

    #[test]
    fn test_priority_policy_lower_priority_blocked() {
        let mut ins = BrandingInserter::new(InsertionPolicy::PriorityWins);
        ins.schedule(bug_slot(1, 0, None, 10)).expect("schedule high-prio");
        ins.schedule(bug_slot(2, 50, None, 3)).expect("schedule low-prio");
        ins.advance_to_frame(0); // activate high-priority slot 1
        let events = ins.advance_to_frame(50);
        // Low-priority slot 2 should NOT produce an Activate event
        let low_activated = events.iter().any(|e| e.slot_id == 2 && e.kind == BrandingEventKind::Activate);
        assert!(!low_activated, "low-priority slot should be blocked");
    }

    #[test]
    fn test_allow_overlap_multiple_zones() {
        let mut ins = BrandingInserter::new(InsertionPolicy::AllowOverlap);
        ins.schedule(bug_slot(1, 0, None, 5)).expect("bug");
        ins.schedule(lower_third_slot(2, 0, None, 5)).expect("lt");
        let events = ins.advance_to_frame(0);
        assert_eq!(events.len(), 2);
        let active = ins.active_slot_ids();
        assert_eq!(active.len(), 2);
    }

    #[test]
    fn test_slots_in_zone_query() {
        let mut ins = BrandingInserter::new(InsertionPolicy::AllowOverlap);
        ins.schedule(bug_slot(1, 0, None, 5)).expect("slot 1");
        ins.schedule(bug_slot(2, 100, Some(200), 3)).expect("slot 2");
        ins.schedule(lower_third_slot(3, 0, None, 5)).expect("lt");
        let bug_slots = ins.slots_in_zone(BrandingZone::BugTopRight);
        assert_eq!(bug_slots.len(), 2);
    }

    #[test]
    fn test_event_log_accumulates() {
        let mut ins = BrandingInserter::new(InsertionPolicy::AllowOverlap);
        ins.schedule(bug_slot(1, 0, Some(50), 5)).expect("schedule");
        ins.advance_to_frame(0);  // activate → 1 event
        ins.advance_to_frame(50); // deactivate → 1 event
        assert_eq!(ins.event_log().len(), 2);
    }
}
