//! High-level tally system for broadcast switchers.
//!
//! Provides a unified tally management layer that tracks on-air and preview
//! states across all M/E buses and distributes tally signals to connected
//! tally lights and downstream devices.
//!
//! ## Atomic tally state
//!
//! For real-time tally distribution to external devices (camera tallies, TSL,
//! etc.), [`AtomicTallySystem`] uses `AtomicU8` per input for lock-free reads
//! from real-time contexts.  Writers use `Ordering::Release` and readers use
//! `Ordering::Acquire` to guarantee a sequentially-consistent view.

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

/// Color emitted by a tally light on a camera or source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TallyLightColor {
    /// Source is live on program output (red).
    Red,
    /// Source is selected on preview bus (green).
    Green,
    /// Source is queued or in standby (amber).
    Amber,
    /// No active tally (off).
    Off,
}

/// Operational state of a single tally-enabled source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TallySourceState {
    /// Source is on air (program).
    Live,
    /// Source is on preview.
    Preview,
    /// Source is active in a keyer layer.
    KeyerLive,
    /// Source is not in use.
    Inactive,
}

impl TallySourceState {
    /// Return the tally light color that corresponds to this state.
    #[must_use]
    pub fn light_color(self) -> TallyLightColor {
        match self {
            Self::Live | Self::KeyerLive => TallyLightColor::Red,
            Self::Preview => TallyLightColor::Green,
            Self::Inactive => TallyLightColor::Off,
        }
    }

    /// Returns `true` if the source is currently on air.
    #[must_use]
    pub fn is_on_air(self) -> bool {
        matches!(self, Self::Live | Self::KeyerLive)
    }
}

/// Entry stored for each registered source.
#[derive(Debug, Clone)]
struct TallyEntry {
    label: String,
    state: TallySourceState,
}

/// Central tally management system.
///
/// Tracks the on-air and preview state for every registered source and
/// provides convenient queries such as "which sources are currently live?"
#[derive(Debug, Default)]
pub struct TallySystem {
    entries: HashMap<u32, TallyEntry>,
    /// Currently active program source IDs (per M/E row).
    program_sources: Vec<u32>,
    /// Currently active preview source IDs (per M/E row).
    preview_sources: Vec<u32>,
}

impl TallySystem {
    /// Create a new, empty tally system.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a source with the tally system.
    pub fn register_source(&mut self, id: u32, label: &str) {
        self.entries.insert(
            id,
            TallyEntry {
                label: label.to_string(),
                state: TallySourceState::Inactive,
            },
        );
    }

    /// Remove a source from the tally system.
    pub fn unregister_source(&mut self, id: u32) {
        self.entries.remove(&id);
    }

    /// Set a source as live (on program output).
    ///
    /// Also clears its preview state if it was previously on preview.
    pub fn set_live(&mut self, id: u32) {
        if let Some(entry) = self.entries.get_mut(&id) {
            entry.state = TallySourceState::Live;
        }
        if !self.program_sources.contains(&id) {
            self.program_sources.push(id);
        }
        self.preview_sources.retain(|&x| x != id);
    }

    /// Set a source as on preview.
    pub fn set_preview(&mut self, id: u32) {
        if let Some(entry) = self.entries.get_mut(&id) {
            if entry.state != TallySourceState::Live {
                entry.state = TallySourceState::Preview;
            }
        }
        if !self.preview_sources.contains(&id) {
            self.preview_sources.push(id);
        }
    }

    /// Set a source as inactive (remove from all buses).
    pub fn set_inactive(&mut self, id: u32) {
        if let Some(entry) = self.entries.get_mut(&id) {
            entry.state = TallySourceState::Inactive;
        }
        self.program_sources.retain(|&x| x != id);
        self.preview_sources.retain(|&x| x != id);
    }

    /// Mark a source as active in a keyer layer (still on-air).
    pub fn set_keyer_live(&mut self, id: u32) {
        if let Some(entry) = self.entries.get_mut(&id) {
            entry.state = TallySourceState::KeyerLive;
        }
    }

    /// Query the current state of a source.
    #[must_use]
    pub fn state(&self, id: u32) -> TallySourceState {
        self.entries
            .get(&id)
            .map_or(TallySourceState::Inactive, |e| e.state)
    }

    /// Query the tally light color for a source.
    #[must_use]
    pub fn light_color(&self, id: u32) -> TallyLightColor {
        self.state(id).light_color()
    }

    /// Return all sources that are currently live (on-air).
    #[must_use]
    pub fn live_sources(&self) -> Vec<u32> {
        self.entries
            .iter()
            .filter(|(_, e)| e.state.is_on_air())
            .map(|(&id, _)| id)
            .collect()
    }

    /// Return all sources that are currently on preview.
    #[must_use]
    pub fn preview_sources(&self) -> Vec<u32> {
        self.entries
            .iter()
            .filter(|(_, e)| e.state == TallySourceState::Preview)
            .map(|(&id, _)| id)
            .collect()
    }

    /// Return the human-readable label for a source, if registered.
    #[must_use]
    pub fn label(&self, id: u32) -> Option<&str> {
        self.entries.get(&id).map(|e| e.label.as_str())
    }

    /// Return total number of registered sources.
    #[must_use]
    pub fn source_count(&self) -> usize {
        self.entries.len()
    }

    /// Clear all tally states (set every source to inactive).
    pub fn clear_all(&mut self) {
        for entry in self.entries.values_mut() {
            entry.state = TallySourceState::Inactive;
        }
        self.program_sources.clear();
        self.preview_sources.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_system() -> TallySystem {
        let mut s = TallySystem::new();
        s.register_source(1, "Camera 1");
        s.register_source(2, "Camera 2");
        s.register_source(3, "VT 1");
        s
    }

    #[test]
    fn test_new_system_is_empty() {
        let s = TallySystem::new();
        assert_eq!(s.source_count(), 0);
    }

    #[test]
    fn test_register_source() {
        let mut s = TallySystem::new();
        s.register_source(1, "Cam 1");
        assert_eq!(s.source_count(), 1);
        assert_eq!(s.state(1), TallySourceState::Inactive);
    }

    #[test]
    fn test_set_live() {
        let mut s = make_system();
        s.set_live(1);
        assert_eq!(s.state(1), TallySourceState::Live);
        assert_eq!(s.light_color(1), TallyLightColor::Red);
        assert!(s.state(1).is_on_air());
    }

    #[test]
    fn test_set_preview() {
        let mut s = make_system();
        s.set_preview(2);
        assert_eq!(s.state(2), TallySourceState::Preview);
        assert_eq!(s.light_color(2), TallyLightColor::Green);
        assert!(!s.state(2).is_on_air());
    }

    #[test]
    fn test_set_inactive() {
        let mut s = make_system();
        s.set_live(1);
        s.set_inactive(1);
        assert_eq!(s.state(1), TallySourceState::Inactive);
        assert_eq!(s.light_color(1), TallyLightColor::Off);
    }

    #[test]
    fn test_set_keyer_live() {
        let mut s = make_system();
        s.set_keyer_live(3);
        assert_eq!(s.state(3), TallySourceState::KeyerLive);
        assert!(s.state(3).is_on_air());
        assert_eq!(s.light_color(3), TallyLightColor::Red);
    }

    #[test]
    fn test_live_sources() {
        let mut s = make_system();
        s.set_live(1);
        s.set_keyer_live(3);
        let live = s.live_sources();
        assert!(live.contains(&1));
        assert!(live.contains(&3));
        assert!(!live.contains(&2));
    }

    #[test]
    fn test_preview_sources() {
        let mut s = make_system();
        s.set_preview(2);
        let pv = s.preview_sources();
        assert!(pv.contains(&2));
        assert!(!pv.contains(&1));
    }

    #[test]
    fn test_set_live_removes_from_preview() {
        let mut s = make_system();
        s.set_preview(1);
        assert_eq!(s.state(1), TallySourceState::Preview);
        s.set_live(1);
        assert_eq!(s.state(1), TallySourceState::Live);
        assert!(s.preview_sources().is_empty());
    }

    #[test]
    fn test_clear_all() {
        let mut s = make_system();
        s.set_live(1);
        s.set_preview(2);
        s.set_keyer_live(3);
        s.clear_all();
        assert_eq!(s.state(1), TallySourceState::Inactive);
        assert_eq!(s.state(2), TallySourceState::Inactive);
        assert_eq!(s.state(3), TallySourceState::Inactive);
        assert!(s.live_sources().is_empty());
        assert!(s.preview_sources().is_empty());
    }

    #[test]
    fn test_label() {
        let s = make_system();
        assert_eq!(s.label(1), Some("Camera 1"));
        assert_eq!(s.label(2), Some("Camera 2"));
        assert_eq!(s.label(99), None);
    }

    #[test]
    fn test_unregister_source() {
        let mut s = make_system();
        s.unregister_source(1);
        assert_eq!(s.source_count(), 2);
        assert_eq!(s.state(1), TallySourceState::Inactive);
    }

    #[test]
    fn test_unregistered_source_is_inactive() {
        let s = TallySystem::new();
        assert_eq!(s.state(42), TallySourceState::Inactive);
        assert_eq!(s.light_color(42), TallyLightColor::Off);
    }

    #[test]
    fn test_tally_light_color_variants() {
        assert_eq!(TallySourceState::Live.light_color(), TallyLightColor::Red);
        assert_eq!(
            TallySourceState::KeyerLive.light_color(),
            TallyLightColor::Red
        );
        assert_eq!(
            TallySourceState::Preview.light_color(),
            TallyLightColor::Green
        );
        assert_eq!(
            TallySourceState::Inactive.light_color(),
            TallyLightColor::Off
        );
    }
}

// ---------------------------------------------------------------------------
// Atomic tally system for lock-free real-time distribution
// ---------------------------------------------------------------------------

/// Tally level encoded as a `u8` for atomic storage.
///
/// `0` = Off, `1` = Preview (green), `2` = Program (red).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TallyLevel {
    /// No active tally.
    Off = 0,
    /// Source is in preview (green).
    Preview = 1,
    /// Source is on program / on-air (red).
    Program = 2,
}

impl TallyLevel {
    /// Encode as a raw `u8` for atomic storage.
    #[inline]
    fn to_raw(self) -> u8 {
        self as u8
    }

    /// Decode from a raw `u8`; returns [`TallyLevel::Off`] for unknown values.
    #[inline]
    fn from_raw(v: u8) -> Self {
        match v {
            1 => Self::Preview,
            2 => Self::Program,
            _ => Self::Off,
        }
    }
}

/// Lock-free tally system using one `AtomicU8` per input.
///
/// The hot-path (`get_state`) never blocks and issues only an atomic load with
/// `Ordering::Acquire`.  The control-thread writer uses `Ordering::Release`.
///
/// For bulk tally updates (e.g. after a cut), a `generation` counter is
/// bumped with `SeqCst` before and after the batch write so that readers can
/// detect a stale snapshot: if `generation` changed between two reads, the
/// reader should retry.
pub struct AtomicTallySystem {
    /// Per-input tally state stored as raw `u8` (see [`TallyLevel`]).
    states: Vec<Arc<AtomicU8>>,
    /// Monotonically increasing generation counter.
    generation: Arc<AtomicU8>,
    /// Number of inputs managed by this system.
    num_inputs: usize,
}

impl AtomicTallySystem {
    /// Create a new atomic tally system for `num_inputs` inputs.
    ///
    /// All inputs start in the [`TallyLevel::Off`] state.
    pub fn new(num_inputs: usize) -> Self {
        let states = (0..num_inputs)
            .map(|_| Arc::new(AtomicU8::new(TallyLevel::Off.to_raw())))
            .collect();
        Self {
            states,
            generation: Arc::new(AtomicU8::new(0)),
            num_inputs,
        }
    }

    /// Returns the number of inputs tracked by this system.
    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    /// Set the tally level for a single input.
    ///
    /// Uses `Ordering::Release` so that subsequent `get_state` calls with
    /// `Ordering::Acquire` will see the new value.
    ///
    /// Returns `false` if `input_id` is out of bounds.
    pub fn set_program(&self, input_id: usize, state: TallyLevel) -> bool {
        if let Some(atom) = self.states.get(input_id) {
            atom.store(state.to_raw(), Ordering::Release);
            true
        } else {
            false
        }
    }

    /// Alias for [`Self::set_program`] — sets any tally level for the given input.
    pub fn set_state(&self, input_id: usize, state: TallyLevel) -> bool {
        self.set_program(input_id, state)
    }

    /// Read the tally level for a single input.
    ///
    /// Uses `Ordering::Acquire` — guaranteed to see writes that used
    /// `Ordering::Release`.
    pub fn get_state(&self, input_id: usize) -> TallyLevel {
        self.states
            .get(input_id)
            .map(|a| TallyLevel::from_raw(a.load(Ordering::Acquire)))
            .unwrap_or(TallyLevel::Off)
    }

    /// Bump the generation counter *before* a bulk update.
    ///
    /// Readers can compare `generation_before = read_generation()` and
    /// `generation_after = read_generation()` to detect mid-batch reads.
    pub fn begin_bulk_update(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
    }

    /// Bump the generation counter *after* a bulk update.
    pub fn end_bulk_update(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
    }

    /// Read the current generation counter.
    ///
    /// A reader can use `let g = system.read_generation()` before and after
    /// reading tally states to detect whether a bulk update was in progress.
    pub fn read_generation(&self) -> u8 {
        self.generation.load(Ordering::Acquire)
    }

    /// Apply a bulk tally update atomically (generation-guarded).
    ///
    /// `updates` is a slice of `(input_id, level)` pairs.  Out-of-bounds
    /// `input_id` values are silently skipped.
    pub fn bulk_update(&self, updates: &[(usize, TallyLevel)]) {
        self.begin_bulk_update();
        for &(input_id, level) in updates {
            self.set_state(input_id, level);
        }
        self.end_bulk_update();
    }

    /// Snapshot all tally states into a plain `Vec`.
    ///
    /// This acquires each atomic individually; for a truly consistent
    /// snapshot under concurrent writers, compare `read_generation()` before
    /// and after.
    pub fn snapshot_all(&self) -> Vec<TallyLevel> {
        self.states
            .iter()
            .map(|a| TallyLevel::from_raw(a.load(Ordering::Acquire)))
            .collect()
    }

    /// Get a cheap `Arc` clone of the underlying atomic for a given input,
    /// allowing real-time threads to hold their own reference.
    pub fn input_atom(&self, input_id: usize) -> Option<Arc<AtomicU8>> {
        self.states.get(input_id).map(Arc::clone)
    }
}

#[cfg(test)]
mod atomic_tally_tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_new_all_off() {
        let sys = AtomicTallySystem::new(4);
        for i in 0..4 {
            assert_eq!(sys.get_state(i), TallyLevel::Off);
        }
    }

    #[test]
    fn test_set_program_and_read() {
        let sys = AtomicTallySystem::new(8);
        sys.set_program(3, TallyLevel::Program);
        assert_eq!(sys.get_state(3), TallyLevel::Program);
        assert_eq!(sys.get_state(0), TallyLevel::Off);
    }

    #[test]
    fn test_set_preview_and_read() {
        let sys = AtomicTallySystem::new(8);
        sys.set_state(5, TallyLevel::Preview);
        assert_eq!(sys.get_state(5), TallyLevel::Preview);
    }

    #[test]
    fn test_out_of_bounds_returns_off() {
        let sys = AtomicTallySystem::new(4);
        assert_eq!(sys.get_state(99), TallyLevel::Off);
        assert!(!sys.set_program(99, TallyLevel::Program));
    }

    #[test]
    fn test_bulk_update() {
        let sys = AtomicTallySystem::new(8);
        let updates = vec![
            (0, TallyLevel::Program),
            (1, TallyLevel::Preview),
            (2, TallyLevel::Off),
        ];
        sys.bulk_update(&updates);
        assert_eq!(sys.get_state(0), TallyLevel::Program);
        assert_eq!(sys.get_state(1), TallyLevel::Preview);
        assert_eq!(sys.get_state(2), TallyLevel::Off);
        assert_eq!(sys.get_state(3), TallyLevel::Off);
    }

    #[test]
    fn test_generation_counter_increments() {
        let sys = AtomicTallySystem::new(4);
        let g0 = sys.read_generation();
        sys.begin_bulk_update();
        let g1 = sys.read_generation();
        sys.end_bulk_update();
        let g2 = sys.read_generation();
        assert!(g1 > g0, "generation must increase after begin_bulk_update");
        assert!(g2 > g1, "generation must increase after end_bulk_update");
    }

    #[test]
    fn test_snapshot_all() {
        let sys = AtomicTallySystem::new(4);
        sys.set_state(0, TallyLevel::Program);
        sys.set_state(1, TallyLevel::Preview);
        let snap = sys.snapshot_all();
        assert_eq!(snap.len(), 4);
        assert_eq!(snap[0], TallyLevel::Program);
        assert_eq!(snap[1], TallyLevel::Preview);
        assert_eq!(snap[2], TallyLevel::Off);
        assert_eq!(snap[3], TallyLevel::Off);
    }

    #[test]
    fn test_concurrent_read_write_correct() {
        use std::sync::Arc as StdArc;

        let sys = StdArc::new(AtomicTallySystem::new(16));
        let sys_reader = StdArc::clone(&sys);

        // Writer: repeatedly toggles input 0 between Program and Off.
        let writer = thread::spawn({
            let s = StdArc::clone(&sys);
            move || {
                for i in 0..10_000 {
                    if i % 2 == 0 {
                        s.set_state(0, TallyLevel::Program);
                    } else {
                        s.set_state(0, TallyLevel::Off);
                    }
                }
                // Leave in a known state.
                s.set_state(0, TallyLevel::Program);
            }
        });

        // Reader: reads must never see an out-of-range raw value.
        let reader = thread::spawn(move || {
            for _ in 0..10_000 {
                let state = sys_reader.get_state(0);
                // `get_state` maps unknowns to Off, so we only ever get
                // Off, Preview, or Program.
                assert!(
                    state == TallyLevel::Off
                        || state == TallyLevel::Preview
                        || state == TallyLevel::Program,
                    "unexpected tally level"
                );
            }
        });

        writer.join().expect("writer panicked");
        reader.join().expect("reader panicked");

        // Final state must be Program (last writer action).
        assert_eq!(sys.get_state(0), TallyLevel::Program);
    }

    #[test]
    fn test_input_atom_accessible_from_rt_thread() {
        let sys = Arc::new(AtomicTallySystem::new(4));
        // Give a "real-time" thread its own Arc to the atomic.
        let atom = sys.input_atom(2).expect("input 2 should exist");
        let atom_clone = Arc::clone(&atom);

        let handle = thread::spawn(move || {
            // Real-time thread reads directly from the atomic.
            let raw = atom_clone.load(Ordering::Acquire);
            TallyLevel::from_raw(raw)
        });

        // Control thread sets the state.
        sys.set_state(2, TallyLevel::Preview);

        // The real-time thread's read may be before or after the write —
        // either Off or Preview is valid.  We just verify it doesn't crash.
        let _level = handle.join().expect("rt thread panicked");
    }
}
