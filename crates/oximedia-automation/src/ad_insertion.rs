//! Dynamic ad break management using SCTE-35 splice commands.
//!
//! This module provides an [`AdBreakManager`] that tracks scheduled ad avails,
//! manages the insertion of SCTE-35 splice signals, and maintains a history of
//! ad breaks for as-run reconciliation.
//!
//! # Architecture
//!
//! ```text
//! AdBreakManager
//!   ├── pending avails  (ordered by scheduled start ms)
//!   ├── active break    (currently in an ad break or None)
//!   └── completed breaks (history for as-run log)
//! ```
//!
//! The manager is deliberately synchronous so it can be polled on every
//! playout tick without locking overhead.  Thread-safety is the caller's
//! responsibility (wrap in `Arc<Mutex<...>>` if needed).

use crate::playlist::scte35::{
    generate_splice_insert, generate_splice_return, SpliceCommand,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use tracing::{info, warn};

// ─────────────────────────────────────────────────────────────────────────────
// Ad avail descriptor
// ─────────────────────────────────────────────────────────────────────────────

/// A single ad break avail (opportunity to insert advertising).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdAvail {
    /// SCTE-35 event identifier (must be unique per channel per session).
    pub event_id: u32,
    /// Scheduled trigger time relative to the start of the current item (ms).
    pub trigger_ms: u64,
    /// Duration of the break in milliseconds.
    pub duration_ms: u64,
    /// Channel this avail belongs to.
    pub channel_id: usize,
    /// Whether the return is automatic (SCTE-35 `auto_return` flag).
    pub auto_return: bool,
    /// Optional label for as-run logging.
    pub label: Option<String>,
}

impl AdAvail {
    /// Create a new ad avail.
    pub fn new(event_id: u32, trigger_ms: u64, duration_ms: u64, channel_id: usize) -> Self {
        Self {
            event_id,
            trigger_ms,
            duration_ms,
            channel_id,
            auto_return: true,
            label: None,
        }
    }

    /// Attach a human-readable label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Break state
// ─────────────────────────────────────────────────────────────────────────────

/// State of an ad break that has started.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveBreak {
    /// The avail that triggered this break.
    pub avail: AdAvail,
    /// Absolute time (ms from item start) when the break started.
    pub started_at_ms: u64,
    /// Absolute time (ms from item start) when the break should end.
    pub ends_at_ms: u64,
    /// The SCTE-35 `splice_insert` command emitted at break start.
    pub splice_in_cmd: SpliceCommand,
}

impl ActiveBreak {
    /// Returns `true` if the break should end at or before `now_ms`.
    pub fn should_end(&self, now_ms: u64) -> bool {
        now_ms >= self.ends_at_ms
    }
}

/// Record of a completed ad break for as-run reconciliation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletedBreak {
    /// Event ID of the SCTE-35 avail.
    pub event_id: u32,
    /// Millisecond position where the break started.
    pub started_at_ms: u64,
    /// Millisecond position where the break ended.
    pub ended_at_ms: u64,
    /// Actual break duration in milliseconds.
    pub actual_duration_ms: u64,
    /// Optional label.
    pub label: Option<String>,
    /// The SCTE-35 `splice_return` command emitted at break end.
    pub splice_out_cmd: SpliceCommand,
}

// ─────────────────────────────────────────────────────────────────────────────
// Manager
// ─────────────────────────────────────────────────────────────────────────────

/// Dynamic ad break manager.
///
/// Maintains a queue of pending [`AdAvail`]s ordered by trigger time, fires
/// SCTE-35 splice signals at the right moment, and tracks the current active
/// break (if any).
#[derive(Debug, Default)]
pub struct AdBreakManager {
    /// Pending avails: `(trigger_ms, event_id)` → avail.
    pending: BTreeMap<(u64, u32), AdAvail>,
    /// Currently active break, if any.
    active: Option<ActiveBreak>,
    /// History of completed breaks.
    completed: Vec<CompletedBreak>,
}

impl AdBreakManager {
    /// Create a new ad break manager.
    pub fn new() -> Self {
        Self::default()
    }

    // ── Registration ─────────────────────────────────────────────────────────

    /// Schedule an ad avail.
    ///
    /// Returns `false` (without replacing) if an avail with the same
    /// `event_id` is already scheduled.
    pub fn schedule(&mut self, avail: AdAvail) -> bool {
        // Reject duplicate event IDs.
        if self.pending.values().any(|a| a.event_id == avail.event_id) {
            warn!(
                "AdBreakManager: duplicate event_id {} ignored",
                avail.event_id
            );
            return false;
        }
        info!(
            "Scheduled ad avail {} at {}ms ({}ms duration) on ch{}",
            avail.event_id, avail.trigger_ms, avail.duration_ms, avail.channel_id
        );
        let key = (avail.trigger_ms, avail.event_id);
        self.pending.insert(key, avail);
        true
    }

    /// Cancel a scheduled avail by `event_id`.  Returns `true` if found.
    pub fn cancel(&mut self, event_id: u32) -> bool {
        let key = self
            .pending
            .keys()
            .find(|(_, eid)| *eid == event_id)
            .cloned();
        if let Some(k) = key {
            self.pending.remove(&k);
            true
        } else {
            false
        }
    }

    // ── Tick ─────────────────────────────────────────────────────────────────

    /// Poll the manager at `now_ms`.
    ///
    /// This method should be called on every playout tick.  It:
    ///
    /// 1. If a break is active and has expired, ends it and returns a
    ///    [`BreakEvent::End`].
    /// 2. If no break is active, checks for a due avail and fires it,
    ///    returning a [`BreakEvent::Start`].
    /// 3. Otherwise returns [`BreakEvent::None`].
    pub fn tick(&mut self, now_ms: u64) -> BreakEvent {
        // ── Check for break end ───────────────────────────────────────────────
        let should_end = self.active.as_ref().map_or(false, |a| a.should_end(now_ms));
        if should_end {
            if let Some(active) = self.active.take() {
                let splice_out = generate_splice_return(active.avail.event_id);
                let completed = CompletedBreak {
                    event_id: active.avail.event_id,
                    started_at_ms: active.started_at_ms,
                    ended_at_ms: now_ms,
                    actual_duration_ms: now_ms.saturating_sub(active.started_at_ms),
                    label: active.avail.label.clone(),
                    splice_out_cmd: splice_out.clone(),
                };
                info!(
                    "Ad break {} ended at {}ms (actual duration {}ms)",
                    completed.event_id, now_ms, completed.actual_duration_ms
                );
                self.completed.push(completed.clone());
                return BreakEvent::End(completed, splice_out);
            }
        }

        // ── Check for break start ─────────────────────────────────────────────
        if self.active.is_none() {
            // Peek at the first pending avail.
            let due_key = self
                .pending
                .keys()
                .next()
                .filter(|(trigger, _)| *trigger <= now_ms)
                .cloned();

            if let Some(key) = due_key {
                if let Some(avail) = self.pending.remove(&key) {
                    // Convert duration_ms → 90kHz ticks for the splice command.
                    let duration_ticks = avail.duration_ms.saturating_mul(90);
                    let splice_in = generate_splice_insert(avail.event_id, 0, duration_ticks);
                    let ends_at_ms = now_ms + avail.duration_ms;
                    info!(
                        "Ad break {} started at {}ms, ends at {}ms",
                        avail.event_id, now_ms, ends_at_ms
                    );
                    let active = ActiveBreak {
                        avail,
                        started_at_ms: now_ms,
                        ends_at_ms,
                        splice_in_cmd: splice_in.clone(),
                    };
                    self.active = Some(active.clone());
                    return BreakEvent::Start(active, splice_in);
                }
            }
        }

        BreakEvent::None
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// Returns `true` if a break is currently in progress.
    pub fn is_in_break(&self) -> bool {
        self.active.is_some()
    }

    /// Return the current active break, if any.
    pub fn active_break(&self) -> Option<&ActiveBreak> {
        self.active.as_ref()
    }

    /// Return the number of pending avails.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Return completed break history.
    pub fn completed_breaks(&self) -> &[CompletedBreak] {
        &self.completed
    }

    /// Clear all pending avails and completed history (e.g. on item change).
    pub fn reset(&mut self) {
        self.pending.clear();
        self.active = None;
        self.completed.clear();
    }
}

/// Events produced by [`AdBreakManager::tick`].
#[derive(Debug)]
pub enum BreakEvent {
    /// No action required this tick.
    None,
    /// A new ad break has started.  Contains the active break state and the
    /// SCTE-35 `splice_insert` command to inject into the transport stream.
    Start(ActiveBreak, SpliceCommand),
    /// The current ad break has ended.  Contains the completed break record
    /// and the SCTE-35 `splice_return` command.
    End(CompletedBreak, SpliceCommand),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn avail(event_id: u32, trigger_ms: u64, duration_ms: u64) -> AdAvail {
        AdAvail::new(event_id, trigger_ms, duration_ms, 0)
    }

    #[test]
    fn test_schedule_and_tick_starts_break() {
        let mut mgr = AdBreakManager::new();
        mgr.schedule(avail(1, 0, 30_000));

        let evt = mgr.tick(0);
        assert!(matches!(evt, BreakEvent::Start(_, _)));
        assert!(mgr.is_in_break());
    }

    #[test]
    fn test_tick_ends_break_after_duration() {
        let mut mgr = AdBreakManager::new();
        mgr.schedule(avail(2, 0, 5_000));

        mgr.tick(0); // start
        assert!(mgr.is_in_break());

        let evt = mgr.tick(5_000); // end
        assert!(matches!(evt, BreakEvent::End(_, _)));
        assert!(!mgr.is_in_break());
        assert_eq!(mgr.completed_breaks().len(), 1);
    }

    #[test]
    fn test_cancel_removes_pending() {
        let mut mgr = AdBreakManager::new();
        mgr.schedule(avail(3, 1000, 5000));
        assert_eq!(mgr.pending_count(), 1);
        assert!(mgr.cancel(3));
        assert_eq!(mgr.pending_count(), 0);
    }

    #[test]
    fn test_duplicate_event_id_rejected() {
        let mut mgr = AdBreakManager::new();
        assert!(mgr.schedule(avail(4, 0, 1000)));
        assert!(!mgr.schedule(avail(4, 500, 2000)), "duplicate should be rejected");
        assert_eq!(mgr.pending_count(), 1);
    }

    #[test]
    fn test_no_event_before_trigger() {
        let mut mgr = AdBreakManager::new();
        mgr.schedule(avail(5, 5000, 1000));

        let evt = mgr.tick(4999);
        assert!(matches!(evt, BreakEvent::None));
        assert!(!mgr.is_in_break());
    }

    #[test]
    fn test_reset_clears_all_state() {
        let mut mgr = AdBreakManager::new();
        mgr.schedule(avail(6, 0, 5000));
        mgr.tick(0);
        mgr.tick(5000);
        assert!(!mgr.completed_breaks().is_empty());
        mgr.reset();
        assert_eq!(mgr.pending_count(), 0);
        assert!(!mgr.is_in_break());
        assert!(mgr.completed_breaks().is_empty());
    }

    #[test]
    fn test_splice_insert_in_start_event() {
        let mut mgr = AdBreakManager::new();
        mgr.schedule(avail(7, 0, 30_000));
        if let BreakEvent::Start(_, cmd) = mgr.tick(0) {
            // First byte of SCTE-35 binary = 0xFC (table_id)
            assert_eq!(cmd.encoded[0], 0xFC);
        } else {
            panic!("Expected Start event");
        }
    }

    #[test]
    fn test_splice_return_in_end_event() {
        let mut mgr = AdBreakManager::new();
        mgr.schedule(avail(8, 0, 1000));
        mgr.tick(0); // start
        if let BreakEvent::End(_, cmd) = mgr.tick(1000) {
            assert_eq!(cmd.encoded[0], 0xFC);
            // out_of_network flag = 0 for return
            assert_eq!(cmd.encoded[8], 0x00);
        } else {
            panic!("Expected End event");
        }
    }
}
