//! GPI (General Purpose Interface) trigger handling for video switchers.
//!
//! GPI inputs are hardware contact-closure or voltage-change signals wired to
//! a switcher's GPIO port.  This module models the full lifecycle:
//!
//! 1. **Edge detection** — Rising, Falling, or Both edges are recognised.
//! 2. **Debounce** — A configurable debounce window (in milliseconds) ignores
//!    glitches immediately after a transition.
//! 3. **Action binding** — Each GPI input can be bound to one or more
//!    [`GpiAction`] variants that are executed when the trigger fires.
//! 4. **Inhibit** — Individual inputs or the entire GPI subsystem can be
//!    inhibited without removing bindings.
//!
//! # Example
//!
//! ```rust
//! use oximedia_switcher::gpi_trigger::{GpiManager, GpiInputConfig, TriggerEdge, GpiAction};
//!
//! let mut manager = GpiManager::new(8);
//!
//! let mut cfg = GpiInputConfig::new(0, "Cam1 Tally");
//! cfg.edge = TriggerEdge::Rising;
//! cfg.debounce_ms = 10;
//! cfg.actions.push(GpiAction::Cut { me_row: 0 });
//! manager.configure(cfg).expect("configure ok");
//!
//! // Feed a HIGH signal (rising edge) — the manager returns fired actions.
//! let actions = manager.feed_signal(0, true, 0).expect("feed ok");
//! assert_eq!(actions.len(), 1);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ────────────────────────────────────────────────────────────────────────────
// Errors
// ────────────────────────────────────────────────────────────────────────────

/// Errors from the GPI trigger subsystem.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum GpiError {
    /// The GPI input index is outside the configured range.
    #[error("GPI input {0} is out of range (max {1})")]
    InputOutOfRange(usize, usize),

    /// No configuration exists for the specified input.
    #[error("GPI input {0} has no configuration")]
    NotConfigured(usize),

    /// A configuration was provided for an out-of-range input.
    #[error("Cannot configure GPI input {0}: out of range (max {1})")]
    ConfigOutOfRange(usize, usize),
}

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

/// Which signal edges cause the trigger to fire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TriggerEdge {
    /// Fire only when the signal goes from LOW to HIGH.
    Rising,
    /// Fire only when the signal goes from HIGH to LOW.
    Falling,
    /// Fire on both rising and falling edges.
    Both,
}

impl TriggerEdge {
    /// Whether this edge type matches the given transition.
    ///
    /// `rising` is `true` if the new signal level is HIGH (was LOW).
    pub fn matches(&self, rising: bool) -> bool {
        match self {
            TriggerEdge::Rising => rising,
            TriggerEdge::Falling => !rising,
            TriggerEdge::Both => true,
        }
    }
}

/// An action to execute when a GPI input fires.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GpiAction {
    /// Perform a cut on the specified M/E row.
    Cut { me_row: usize },
    /// Perform an auto-transition on the specified M/E row.
    AutoTransition { me_row: usize },
    /// Select a program source on the specified M/E row.
    SelectProgram { me_row: usize, input: usize },
    /// Select a preview source on the specified M/E row.
    SelectPreview { me_row: usize, input: usize },
    /// Toggle fade-to-black on the specified M/E row.
    FadeToBlack { me_row: usize },
    /// Run the macro with the given ID.
    RunMacro { macro_id: usize },
    /// Toggle a keyer on or off.
    ToggleKeyer { keyer_id: usize },
    /// Select an aux output source.
    SelectAux { aux_id: usize, input: usize },
    /// Custom action identified by a string tag (for extensibility).
    Custom { tag: String },
}

/// Current logical level of a GPI input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalLevel {
    Low,
    High,
}

impl From<bool> for SignalLevel {
    fn from(high: bool) -> Self {
        if high {
            SignalLevel::High
        } else {
            SignalLevel::Low
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Input configuration
// ────────────────────────────────────────────────────────────────────────────

/// Per-input GPI configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpiInputConfig {
    /// Zero-based GPI input index.
    pub index: usize,
    /// Human-readable label for this input.
    pub label: String,
    /// Which signal edges fire the trigger.
    pub edge: TriggerEdge,
    /// Debounce window in milliseconds.  Subsequent transitions within this
    /// window after a valid trigger are ignored.
    pub debounce_ms: u64,
    /// Actions to execute when the trigger fires.
    pub actions: Vec<GpiAction>,
    /// Whether this input is inhibited (bindings preserved but trigger silent).
    pub inhibited: bool,
}

impl GpiInputConfig {
    /// Create a new configuration with safe defaults.
    pub fn new(index: usize, label: impl Into<String>) -> Self {
        Self {
            index,
            label: label.into(),
            edge: TriggerEdge::Rising,
            debounce_ms: 5,
            actions: Vec::new(),
            inhibited: false,
        }
    }

    /// Add an action to this input.
    pub fn add_action(&mut self, action: GpiAction) {
        self.actions.push(action);
    }

    /// Remove all actions from this input.
    pub fn clear_actions(&mut self) {
        self.actions.clear();
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Runtime state per input
// ────────────────────────────────────────────────────────────────────────────

/// Internal runtime state for a single GPI input.
struct GpiInputState {
    /// Current signal level.
    current_level: SignalLevel,
    /// Timestamp (ms) of the last time the trigger fired.
    last_fired_ms: Option<u64>,
}

impl GpiInputState {
    fn new() -> Self {
        Self {
            current_level: SignalLevel::Low,
            last_fired_ms: None,
        }
    }

    /// Returns `true` if a new signal is a valid edge transition.
    fn is_edge(&self, new_level: SignalLevel) -> bool {
        new_level != self.current_level
    }

    /// Returns `true` if the debounce window has elapsed.
    fn debounce_clear(&self, now_ms: u64, debounce_ms: u64) -> bool {
        match self.last_fired_ms {
            None => true,
            Some(last) => now_ms.saturating_sub(last) >= debounce_ms,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// GpiManager
// ────────────────────────────────────────────────────────────────────────────

/// Manages all GPI inputs: configuration, debounce state, and action dispatch.
pub struct GpiManager {
    /// Maximum number of GPI inputs supported.
    input_count: usize,
    /// Per-input configurations, keyed by index.
    configs: HashMap<usize, GpiInputConfig>,
    /// Per-input runtime state.
    states: Vec<GpiInputState>,
    /// Whether the entire GPI subsystem is inhibited.
    global_inhibit: bool,
}

impl GpiManager {
    /// Create a manager for `input_count` GPI inputs.
    pub fn new(input_count: usize) -> Self {
        let states = (0..input_count).map(|_| GpiInputState::new()).collect();
        Self {
            input_count,
            configs: HashMap::new(),
            states,
            global_inhibit: false,
        }
    }

    // ── Configuration ────────────────────────────────────────────────────────

    /// Register or replace the configuration for a GPI input.
    pub fn configure(&mut self, config: GpiInputConfig) -> Result<(), GpiError> {
        if config.index >= self.input_count {
            return Err(GpiError::ConfigOutOfRange(config.index, self.input_count));
        }
        self.configs.insert(config.index, config);
        Ok(())
    }

    /// Remove the configuration for a GPI input (stops it firing).
    pub fn remove_config(&mut self, index: usize) -> Result<(), GpiError> {
        if index >= self.input_count {
            return Err(GpiError::InputOutOfRange(index, self.input_count));
        }
        self.configs.remove(&index);
        Ok(())
    }

    /// Get an immutable reference to an input configuration.
    pub fn get_config(&self, index: usize) -> Option<&GpiInputConfig> {
        self.configs.get(&index)
    }

    /// Get a mutable reference to an input configuration.
    pub fn get_config_mut(&mut self, index: usize) -> Option<&mut GpiInputConfig> {
        self.configs.get_mut(&index)
    }

    // ── Inhibit control ──────────────────────────────────────────────────────

    /// Set the global inhibit flag.  When `true` no GPI triggers fire.
    pub fn set_global_inhibit(&mut self, inhibit: bool) {
        self.global_inhibit = inhibit;
    }

    /// Whether global inhibit is active.
    pub fn is_globally_inhibited(&self) -> bool {
        self.global_inhibit
    }

    /// Inhibit a specific GPI input without altering its bindings.
    pub fn inhibit_input(&mut self, index: usize) -> Result<(), GpiError> {
        self.configs
            .get_mut(&index)
            .ok_or(GpiError::NotConfigured(index))
            .map(|c| {
                c.inhibited = true;
            })
    }

    /// Un-inhibit a specific GPI input.
    pub fn uninhibit_input(&mut self, index: usize) -> Result<(), GpiError> {
        self.configs
            .get_mut(&index)
            .ok_or(GpiError::NotConfigured(index))
            .map(|c| {
                c.inhibited = false;
            })
    }

    // ── Signal feed ──────────────────────────────────────────────────────────

    /// Feed a new signal level for GPI input `index` at time `now_ms`.
    ///
    /// Returns a (possibly empty) list of [`GpiAction`]s that should be
    /// executed by the caller.  The caller is responsible for actually
    /// performing those actions on the switcher.
    ///
    /// # Parameters
    ///
    /// - `index`   — zero-based GPI input index.
    /// - `high`    — `true` = HIGH level, `false` = LOW level.
    /// - `now_ms`  — current monotonic time in milliseconds (used for
    ///               debounce; pass the switcher frame timestamp).
    pub fn feed_signal(
        &mut self,
        index: usize,
        high: bool,
        now_ms: u64,
    ) -> Result<Vec<GpiAction>, GpiError> {
        if index >= self.input_count {
            return Err(GpiError::InputOutOfRange(index, self.input_count));
        }

        // If globally inhibited, just update the level and return.
        if self.global_inhibit {
            self.states[index].current_level = SignalLevel::from(high);
            return Ok(Vec::new());
        }

        let config = match self.configs.get(&index) {
            Some(c) => c,
            None => {
                self.states[index].current_level = SignalLevel::from(high);
                return Ok(Vec::new());
            }
        };

        let new_level = SignalLevel::from(high);
        let state = &self.states[index];

        // Not an edge transition → no trigger.
        if !state.is_edge(new_level) {
            return Ok(Vec::new());
        }

        let is_rising = new_level == SignalLevel::High;

        // Edge type doesn't match config → update level, no trigger.
        if !config.edge.matches(is_rising) {
            self.states[index].current_level = new_level;
            return Ok(Vec::new());
        }

        // Input inhibited → update level, no trigger.
        if config.inhibited {
            self.states[index].current_level = new_level;
            return Ok(Vec::new());
        }

        // Within debounce window → update level, no trigger.
        let debounce_ms = config.debounce_ms;
        if !state.debounce_clear(now_ms, debounce_ms) {
            self.states[index].current_level = new_level;
            return Ok(Vec::new());
        }

        // Trigger fires: collect actions, update state.
        let fired_actions = config.actions.clone();
        let state_mut = &mut self.states[index];
        state_mut.current_level = new_level;
        state_mut.last_fired_ms = Some(now_ms);

        Ok(fired_actions)
    }

    /// Force the current signal level for an input without triggering debounce
    /// or actions (useful for initialisation).
    pub fn set_initial_level(&mut self, index: usize, high: bool) -> Result<(), GpiError> {
        if index >= self.input_count {
            return Err(GpiError::InputOutOfRange(index, self.input_count));
        }
        self.states[index].current_level = SignalLevel::from(high);
        Ok(())
    }

    /// Current signal level for an input.
    pub fn current_level(&self, index: usize) -> Option<SignalLevel> {
        self.states.get(index).map(|s| s.current_level)
    }

    /// Number of configured GPI inputs.
    pub fn config_count(&self) -> usize {
        self.configs.len()
    }

    /// Total number of GPI inputs (capacity).
    pub fn input_count(&self) -> usize {
        self.input_count
    }

    /// All configured input indices.
    pub fn configured_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = self.configs.keys().copied().collect();
        indices.sort_unstable();
        indices
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn basic_manager() -> GpiManager {
        GpiManager::new(8)
    }

    fn make_config(index: usize, edge: TriggerEdge, actions: Vec<GpiAction>) -> GpiInputConfig {
        let mut cfg = GpiInputConfig::new(index, format!("GPI{index}"));
        cfg.edge = edge;
        cfg.debounce_ms = 0; // disable debounce in most tests
        cfg.actions = actions;
        cfg
    }

    #[test]
    fn test_rising_edge_fires() {
        let mut mgr = basic_manager();
        mgr.configure(make_config(
            0,
            TriggerEdge::Rising,
            vec![GpiAction::Cut { me_row: 0 }],
        ))
        .expect("configure ok");

        let actions = mgr.feed_signal(0, true, 0).expect("feed ok");
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0], GpiAction::Cut { me_row: 0 });
    }

    #[test]
    fn test_falling_edge_fires() {
        let mut mgr = basic_manager();
        // Pre-set level to HIGH so falling is a real edge.
        mgr.set_initial_level(0, true).expect("init ok");
        mgr.configure(make_config(
            0,
            TriggerEdge::Falling,
            vec![GpiAction::FadeToBlack { me_row: 0 }],
        ))
        .expect("configure ok");

        let actions = mgr.feed_signal(0, false, 0).expect("feed ok");
        assert_eq!(actions.len(), 1);
        assert!(matches!(actions[0], GpiAction::FadeToBlack { me_row: 0 }));
    }

    #[test]
    fn test_both_edges_fire() {
        let mut mgr = basic_manager();
        mgr.configure(make_config(
            0,
            TriggerEdge::Both,
            vec![GpiAction::AutoTransition { me_row: 0 }],
        ))
        .expect("configure ok");

        // Rising edge.
        let a1 = mgr.feed_signal(0, true, 0).expect("feed 1");
        assert_eq!(a1.len(), 1);

        // Falling edge.
        let a2 = mgr.feed_signal(0, false, 0).expect("feed 2");
        assert_eq!(a2.len(), 1);
    }

    #[test]
    fn test_same_level_no_fire() {
        let mut mgr = basic_manager();
        mgr.configure(make_config(
            0,
            TriggerEdge::Rising,
            vec![GpiAction::Cut { me_row: 0 }],
        ))
        .expect("configure ok");

        mgr.feed_signal(0, true, 0).expect("feed 1");
        // Same HIGH level again — not an edge.
        let a = mgr.feed_signal(0, true, 0).expect("feed 2");
        assert!(a.is_empty());
    }

    #[test]
    fn test_debounce_suppresses_rapid_retriggering() {
        let mut mgr = basic_manager();
        let mut cfg = GpiInputConfig::new(0, "D");
        cfg.edge = TriggerEdge::Both;
        cfg.debounce_ms = 50; // 50 ms debounce
        cfg.actions.push(GpiAction::Cut { me_row: 0 });
        mgr.configure(cfg).expect("configure ok");

        // First trigger fires at t=0.
        let a1 = mgr.feed_signal(0, true, 0).expect("t0");
        assert_eq!(a1.len(), 1);

        // Retrigger at t=10 ms (within debounce) — suppressed.
        let a2 = mgr.feed_signal(0, false, 10).expect("t10");
        assert!(a2.is_empty(), "debounce should suppress");

        // Retrigger at t=60 ms (after debounce) — fires.
        let a3 = mgr.feed_signal(0, true, 60).expect("t60");
        assert_eq!(a3.len(), 1);
    }

    #[test]
    fn test_unconfigured_input_returns_empty() {
        let mut mgr = basic_manager();
        // Input 3 has no config.
        let actions = mgr.feed_signal(3, true, 0).expect("feed ok");
        assert!(actions.is_empty());
    }

    #[test]
    fn test_out_of_range_error() {
        let mut mgr = basic_manager();
        let err = mgr.feed_signal(99, true, 0).expect_err("out of range");
        assert!(matches!(err, GpiError::InputOutOfRange(99, 8)));
    }

    #[test]
    fn test_global_inhibit() {
        let mut mgr = basic_manager();
        mgr.configure(make_config(
            0,
            TriggerEdge::Rising,
            vec![GpiAction::Cut { me_row: 0 }],
        ))
        .expect("configure ok");

        mgr.set_global_inhibit(true);
        let actions = mgr.feed_signal(0, true, 0).expect("feed ok");
        assert!(actions.is_empty(), "global inhibit should suppress");

        mgr.set_global_inhibit(false);
        // The level is already HIGH due to the inhibited feed, so feed LOW→HIGH.
        let _ = mgr.feed_signal(0, false, 1); // reset to LOW
        let actions = mgr.feed_signal(0, true, 2).expect("feed ok");
        assert_eq!(actions.len(), 1, "should fire after inhibit lifted");
    }

    #[test]
    fn test_per_input_inhibit() {
        let mut mgr = basic_manager();
        mgr.configure(make_config(
            0,
            TriggerEdge::Rising,
            vec![GpiAction::Cut { me_row: 0 }],
        ))
        .expect("configure ok");

        mgr.inhibit_input(0).expect("inhibit ok");
        let actions = mgr.feed_signal(0, true, 0).expect("feed ok");
        assert!(actions.is_empty());

        mgr.uninhibit_input(0).expect("uninhibit ok");
        let _ = mgr.feed_signal(0, false, 1); // reset level
        let actions = mgr.feed_signal(0, true, 2).expect("feed ok");
        assert_eq!(actions.len(), 1);
    }

    #[test]
    fn test_multiple_actions_on_single_trigger() {
        let mut mgr = basic_manager();
        let actions = vec![
            GpiAction::Cut { me_row: 0 },
            GpiAction::SelectAux {
                aux_id: 0,
                input: 5,
            },
            GpiAction::RunMacro { macro_id: 3 },
        ];
        mgr.configure(make_config(0, TriggerEdge::Rising, actions))
            .expect("configure ok");

        let fired = mgr.feed_signal(0, true, 0).expect("feed ok");
        assert_eq!(fired.len(), 3);
    }

    #[test]
    fn test_configure_replaces_existing() {
        let mut mgr = basic_manager();
        mgr.configure(make_config(
            0,
            TriggerEdge::Rising,
            vec![GpiAction::Cut { me_row: 0 }],
        ))
        .expect("first configure ok");

        // Replace with a different configuration.
        mgr.configure(make_config(
            0,
            TriggerEdge::Falling,
            vec![GpiAction::FadeToBlack { me_row: 0 }],
        ))
        .expect("second configure ok");

        // Rising edge should NOT fire (now configured for Falling).
        let a = mgr.feed_signal(0, true, 0).expect("feed ok");
        assert!(a.is_empty(), "replaced config: rising should not fire");
    }

    #[test]
    fn test_remove_config() {
        let mut mgr = basic_manager();
        mgr.configure(make_config(
            0,
            TriggerEdge::Rising,
            vec![GpiAction::Cut { me_row: 0 }],
        ))
        .expect("configure ok");
        assert_eq!(mgr.config_count(), 1);

        mgr.remove_config(0).expect("remove ok");
        assert_eq!(mgr.config_count(), 0);

        // After removal, feed returns empty (no config).
        let a = mgr.feed_signal(0, true, 0).expect("feed ok");
        assert!(a.is_empty());
    }

    #[test]
    fn test_trigger_edge_matches() {
        assert!(TriggerEdge::Rising.matches(true));
        assert!(!TriggerEdge::Rising.matches(false));
        assert!(!TriggerEdge::Falling.matches(true));
        assert!(TriggerEdge::Falling.matches(false));
        assert!(TriggerEdge::Both.matches(true));
        assert!(TriggerEdge::Both.matches(false));
    }

    #[test]
    fn test_configured_indices_sorted() {
        let mut mgr = basic_manager();
        mgr.configure(make_config(5, TriggerEdge::Rising, vec![]))
            .expect("cfg 5");
        mgr.configure(make_config(1, TriggerEdge::Rising, vec![]))
            .expect("cfg 1");
        mgr.configure(make_config(3, TriggerEdge::Rising, vec![]))
            .expect("cfg 3");

        let indices = mgr.configured_indices();
        assert_eq!(indices, vec![1, 3, 5]);
    }

    #[test]
    fn test_current_level_tracking() {
        let mut mgr = basic_manager();
        mgr.configure(make_config(2, TriggerEdge::Both, vec![]))
            .expect("cfg");

        assert_eq!(mgr.current_level(2), Some(SignalLevel::Low));
        mgr.feed_signal(2, true, 0).expect("feed");
        assert_eq!(mgr.current_level(2), Some(SignalLevel::High));
        mgr.feed_signal(2, false, 0).expect("feed");
        assert_eq!(mgr.current_level(2), Some(SignalLevel::Low));
    }
}
