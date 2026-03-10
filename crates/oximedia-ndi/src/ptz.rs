//! PTZ (Pan-Tilt-Zoom) command model for `oximedia-ndi`.
//!
//! Provides typed PTZ commands, a preset store, and a command sequencer that
//! validates speed values before queuing them.

#![allow(dead_code)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]

// ---------------------------------------------------------------------------
// PtzAxis
// ---------------------------------------------------------------------------

/// Which camera axis a movement command addresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtzAxis {
    /// Horizontal rotation.
    Pan,
    /// Vertical rotation.
    Tilt,
    /// Optical / digital zoom.
    Zoom,
    /// Lens focus.
    Focus,
}

// ---------------------------------------------------------------------------
// PtzCommand
// ---------------------------------------------------------------------------

/// A single PTZ command to be sent to a camera.
#[derive(Debug, Clone, PartialEq)]
pub enum PtzCommand {
    /// Move on `axis` at `speed` (−1.0 … +1.0; negative = reverse).
    Move {
        /// The camera axis to move.
        axis: PtzAxis,
        /// Movement speed clamped to [−1.0, 1.0].
        speed: f32,
    },
    /// Stop all motion immediately.
    Stop,
    /// Store current position as preset `index`.
    StorePreset(u8),
    /// Recall a previously stored preset.
    RecallPreset(u8),
    /// Request the camera to auto-focus.
    AutoFocus,
}

impl PtzCommand {
    /// Returns `true` when the command results in camera motion.
    pub fn causes_motion(&self) -> bool {
        matches!(self, Self::Move { .. } | Self::RecallPreset(_))
    }

    /// Speed magnitude in [0.0, 1.0] for [`PtzCommand::Move`]; `None` otherwise.
    pub fn speed_magnitude(&self) -> Option<f32> {
        if let Self::Move { speed, .. } = self {
            Some(speed.abs())
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// PtzPresetStore
// ---------------------------------------------------------------------------

/// Stores up to 256 named PTZ presets (indexed 0 – 255).
#[derive(Debug, Clone)]
pub struct PtzPresetStore {
    /// Stored preset labels; `None` when the slot is empty.
    pub slots: [Option<String>; 256],
}

impl Default for PtzPresetStore {
    fn default() -> Self {
        // Can't derive Default because [Option<String>; 256] lacks a Default impl.
        // SAFETY: None is a valid zero-initialisation for Option<String>.
        Self {
            slots: std::array::from_fn(|_| None),
        }
    }
}

impl PtzPresetStore {
    /// Create an empty preset store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Store `label` at `index`.  Overwrites any previous entry.
    pub fn store(&mut self, index: u8, label: &str) {
        self.slots[usize::from(index)] = Some(label.to_string());
    }

    /// Retrieve the label for `index`, if any.
    pub fn recall(&self, index: u8) -> Option<&str> {
        self.slots[usize::from(index)].as_deref()
    }

    /// Clear the preset at `index`.  Returns `true` when there was one.
    pub fn clear(&mut self, index: u8) -> bool {
        let slot = &mut self.slots[usize::from(index)];
        let had = slot.is_some();
        *slot = None;
        had
    }

    /// Number of occupied preset slots.
    pub fn count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }
}

// ---------------------------------------------------------------------------
// PtzCommandQueue
// ---------------------------------------------------------------------------

/// Validates and queues PTZ commands, rejecting those with out-of-range speed.
#[derive(Debug, Clone, Default)]
pub struct PtzCommandQueue {
    /// Queued commands (oldest first).
    pub queue: Vec<PtzCommand>,
    /// Number of commands rejected due to invalid speed.
    pub rejected: u64,
}

impl PtzCommandQueue {
    /// Create an empty command queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enqueue `cmd`, validating speed for [`PtzCommand::Move`].
    ///
    /// Returns `false` when the command was rejected (speed > 1.0).
    pub fn push(&mut self, cmd: PtzCommand) -> bool {
        if let PtzCommand::Move { speed, .. } = &cmd {
            if speed.abs() > 1.0 {
                self.rejected += 1;
                return false;
            }
        }
        self.queue.push(cmd);
        true
    }

    /// Remove and return the oldest queued command.
    pub fn pop(&mut self) -> Option<PtzCommand> {
        if self.queue.is_empty() {
            None
        } else {
            Some(self.queue.remove(0))
        }
    }

    /// Number of commands currently in the queue.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Returns `true` when the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ptz_axis_variants() {
        let axes = [PtzAxis::Pan, PtzAxis::Tilt, PtzAxis::Zoom, PtzAxis::Focus];
        assert_eq!(axes.len(), 4);
    }

    #[test]
    fn test_command_causes_motion_move() {
        let cmd = PtzCommand::Move {
            axis: PtzAxis::Pan,
            speed: 0.5,
        };
        assert!(cmd.causes_motion());
    }

    #[test]
    fn test_command_causes_motion_recall() {
        assert!(PtzCommand::RecallPreset(1).causes_motion());
    }

    #[test]
    fn test_command_causes_motion_stop_false() {
        assert!(!PtzCommand::Stop.causes_motion());
    }

    #[test]
    fn test_command_causes_motion_auto_focus_false() {
        assert!(!PtzCommand::AutoFocus.causes_motion());
    }

    #[test]
    fn test_speed_magnitude_move() {
        let cmd = PtzCommand::Move {
            axis: PtzAxis::Tilt,
            speed: -0.75,
        };
        assert!((cmd.speed_magnitude().expect("expected speed magnitude") - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_speed_magnitude_non_move_is_none() {
        assert_eq!(PtzCommand::Stop.speed_magnitude(), None);
    }

    #[test]
    fn test_preset_store_store_recall() {
        let mut store = PtzPresetStore::new();
        store.store(5, "home");
        assert_eq!(store.recall(5), Some("home"));
    }

    #[test]
    fn test_preset_store_recall_empty() {
        let store = PtzPresetStore::new();
        assert_eq!(store.recall(0), None);
    }

    #[test]
    fn test_preset_store_clear() {
        let mut store = PtzPresetStore::new();
        store.store(3, "studio");
        assert!(store.clear(3));
        assert_eq!(store.recall(3), None);
    }

    #[test]
    fn test_preset_store_clear_empty_returns_false() {
        let mut store = PtzPresetStore::new();
        assert!(!store.clear(10));
    }

    #[test]
    fn test_preset_store_count() {
        let mut store = PtzPresetStore::new();
        store.store(0, "a");
        store.store(1, "b");
        assert_eq!(store.count(), 2);
    }

    #[test]
    fn test_command_queue_push_and_pop() {
        let mut q = PtzCommandQueue::new();
        q.push(PtzCommand::Stop);
        assert!(!q.is_empty());
        assert_eq!(q.pop(), Some(PtzCommand::Stop));
        assert!(q.is_empty());
    }

    #[test]
    fn test_command_queue_rejects_high_speed() {
        let mut q = PtzCommandQueue::new();
        let accepted = q.push(PtzCommand::Move {
            axis: PtzAxis::Zoom,
            speed: 1.5,
        });
        assert!(!accepted);
        assert_eq!(q.rejected, 1);
        assert!(q.is_empty());
    }

    #[test]
    fn test_command_queue_accepts_boundary_speed() {
        let mut q = PtzCommandQueue::new();
        let ok = q.push(PtzCommand::Move {
            axis: PtzAxis::Focus,
            speed: 1.0,
        });
        assert!(ok);
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn test_command_queue_fifo_order() {
        let mut q = PtzCommandQueue::new();
        q.push(PtzCommand::StorePreset(0));
        q.push(PtzCommand::RecallPreset(0));
        assert_eq!(q.pop(), Some(PtzCommand::StorePreset(0)));
        assert_eq!(q.pop(), Some(PtzCommand::RecallPreset(0)));
    }
}
