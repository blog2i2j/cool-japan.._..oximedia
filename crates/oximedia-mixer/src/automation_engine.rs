//! Sample-accurate automation playback engine.
//!
//! This module provides an [`AutomationEngine`] that reads automation lane data
//! and applies parameter changes per audio buffer with sample-accurate timing.
//!
//! ## Modes
//!
//! - **Read**: Play back recorded automation — breakpoints are evaluated each
//!   buffer and interpolated values are emitted.
//! - **Touch**: Automation is overridden while the user is actively touching a
//!   control; once released, playback snaps back to the recorded curve.
//! - **Latch**: Like Touch, but the last manual value is held (latched) even
//!   after the user releases the control, overriding the recorded curve until
//!   playback is stopped/reset.
//!
//! ## Interpolation
//!
//! Linear interpolation is performed between consecutive
//! [`AutomationBreakpoint`]s.  Values before the first or after the last
//! breakpoint hold the nearest breakpoint value.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// AutomationMode
// ---------------------------------------------------------------------------

/// Automation playback mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AutomationMode {
    /// Play back recorded breakpoints with linear interpolation.
    Read,
    /// Override while touching; revert to recorded curve on release.
    Touch,
    /// Override while touching; hold last value after release.
    Latch,
}

// ---------------------------------------------------------------------------
// AutomationBreakpoint
// ---------------------------------------------------------------------------

/// A single automation keyframe that maps an absolute sample position to a
/// parameter value.
#[derive(Debug, Clone, PartialEq)]
pub struct AutomationBreakpoint {
    /// Absolute sample position (relative to timeline start).
    pub sample_pos: u64,
    /// Parameter value at this position.
    pub value: f32,
}

impl AutomationBreakpoint {
    /// Create a new breakpoint.
    #[must_use]
    pub fn new(sample_pos: u64, value: f32) -> Self {
        Self { sample_pos, value }
    }
}

// ---------------------------------------------------------------------------
// AutomationLaneData
// ---------------------------------------------------------------------------

/// Stores breakpoints for a single automated parameter.
///
/// Breakpoints are kept sorted by `sample_pos` internally.
#[derive(Debug, Clone)]
pub struct AutomationLaneData {
    /// Unique string identifier for the parameter (e.g. `"ch1_gain"`).
    pub parameter_id: String,
    /// Minimum allowed value (output is clamped).
    pub min_value: f32,
    /// Maximum allowed value (output is clamped).
    pub max_value: f32,
    /// Sorted breakpoints.
    breakpoints: Vec<AutomationBreakpoint>,
}

impl AutomationLaneData {
    /// Create a new lane with no breakpoints.
    #[must_use]
    pub fn new(parameter_id: impl Into<String>, min_value: f32, max_value: f32) -> Self {
        Self {
            parameter_id: parameter_id.into(),
            min_value,
            max_value,
            breakpoints: Vec::new(),
        }
    }

    /// Add a breakpoint, maintaining sorted order by `sample_pos`.
    pub fn add_breakpoint(&mut self, bp: AutomationBreakpoint) {
        let pos = self
            .breakpoints
            .partition_point(|b| b.sample_pos <= bp.sample_pos);
        self.breakpoints.insert(pos, bp);
    }

    /// Number of breakpoints.
    #[must_use]
    pub fn breakpoint_count(&self) -> usize {
        self.breakpoints.len()
    }

    /// Access breakpoints slice.
    #[must_use]
    pub fn breakpoints(&self) -> &[AutomationBreakpoint] {
        &self.breakpoints
    }

    /// Evaluate the lane at an absolute sample position with linear interpolation.
    #[must_use]
    pub fn evaluate(&self, sample_pos: u64) -> f32 {
        if self.breakpoints.is_empty() {
            return (self.min_value + self.max_value) * 0.5;
        }

        let first = &self.breakpoints[0];
        if sample_pos <= first.sample_pos {
            return first.value.clamp(self.min_value, self.max_value);
        }

        let last = &self.breakpoints[self.breakpoints.len() - 1];
        if sample_pos >= last.sample_pos {
            return last.value.clamp(self.min_value, self.max_value);
        }

        // Binary search for the segment containing `sample_pos`.
        let idx = self
            .breakpoints
            .partition_point(|b| b.sample_pos <= sample_pos)
            .saturating_sub(1);

        let p0 = &self.breakpoints[idx];
        let p1 = &self.breakpoints[idx + 1];

        let span = p1.sample_pos.saturating_sub(p0.sample_pos);
        let raw = if span == 0 {
            p1.value
        } else {
            let t = (sample_pos.saturating_sub(p0.sample_pos)) as f64 / span as f64;
            let t = t as f32;
            p0.value + t * (p1.value - p0.value)
        };

        raw.clamp(self.min_value, self.max_value)
    }

    /// Clear all breakpoints.
    pub fn clear(&mut self) {
        self.breakpoints.clear();
    }
}

// ---------------------------------------------------------------------------
// LaneState (per-lane runtime state)
// ---------------------------------------------------------------------------

/// Per-lane runtime state for Touch/Latch overrides.
#[derive(Debug, Clone)]
struct LaneState {
    /// The automation mode for this lane.
    mode: AutomationMode,
    /// Whether the user is currently "touching" the control.
    touching: bool,
    /// Manual override value (used in Touch/Latch while touching, or Latch
    /// after release).
    manual_value: Option<f32>,
    /// Whether latch is active (user has touched and released).
    latched: bool,
}

impl LaneState {
    fn new(mode: AutomationMode) -> Self {
        Self {
            mode,
            touching: false,
            manual_value: None,
            latched: false,
        }
    }
}

// ---------------------------------------------------------------------------
// AutomationEngine
// ---------------------------------------------------------------------------

/// Sample-accurate automation playback engine.
///
/// The engine holds multiple [`AutomationLaneData`] instances and, for each
/// processing block, renders per-sample parameter values that the mixer can
/// apply directly.
#[derive(Debug, Clone)]
pub struct AutomationEngine {
    /// Registered lanes keyed by parameter ID.
    lanes: HashMap<String, AutomationLaneData>,
    /// Per-lane runtime state.
    states: HashMap<String, LaneState>,
    /// Rendered per-sample values for the most recent block.
    rendered: HashMap<String, Vec<f32>>,
    /// Global automation mode (used as default for new lanes).
    global_mode: AutomationMode,
    /// Whether the engine is enabled.
    enabled: bool,
}

impl AutomationEngine {
    /// Create a new engine.
    #[must_use]
    pub fn new(mode: AutomationMode) -> Self {
        Self {
            lanes: HashMap::new(),
            states: HashMap::new(),
            rendered: HashMap::new(),
            global_mode: mode,
            enabled: true,
        }
    }

    /// Set the global automation mode.
    pub fn set_mode(&mut self, mode: AutomationMode) {
        self.global_mode = mode;
        for state in self.states.values_mut() {
            state.mode = mode;
        }
    }

    /// Current global mode.
    #[must_use]
    pub fn mode(&self) -> AutomationMode {
        self.global_mode
    }

    /// Enable or disable the engine.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Whether the engine is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Register an automation lane.  Replaces any existing lane with the same
    /// parameter ID.
    pub fn add_lane(&mut self, lane: AutomationLaneData) {
        let id = lane.parameter_id.clone();
        self.states
            .entry(id.clone())
            .or_insert_with(|| LaneState::new(self.global_mode));
        self.lanes.insert(id, lane);
    }

    /// Remove a lane by parameter ID.
    pub fn remove_lane(&mut self, parameter_id: &str) {
        self.lanes.remove(parameter_id);
        self.states.remove(parameter_id);
        self.rendered.remove(parameter_id);
    }

    /// Number of registered lanes.
    #[must_use]
    pub fn lane_count(&self) -> usize {
        self.lanes.len()
    }

    /// Set the automation mode for a single lane.
    pub fn set_lane_mode(&mut self, parameter_id: &str, mode: AutomationMode) {
        if let Some(state) = self.states.get_mut(parameter_id) {
            state.mode = mode;
        }
    }

    // -- Touch / Latch controls ---------------------------------------------

    /// Begin touching a parameter (Touch/Latch override).
    pub fn touch_begin(&mut self, parameter_id: &str, value: f32) {
        if let Some(state) = self.states.get_mut(parameter_id) {
            state.touching = true;
            state.manual_value = Some(value);
        }
    }

    /// Update the manual value while touching.
    pub fn touch_update(&mut self, parameter_id: &str, value: f32) {
        if let Some(state) = self.states.get_mut(parameter_id) {
            if state.touching {
                state.manual_value = Some(value);
            }
        }
    }

    /// End touch.
    ///
    /// In Touch mode the manual override is cleared.  In Latch mode the last
    /// value is held.
    pub fn touch_end(&mut self, parameter_id: &str) {
        if let Some(state) = self.states.get_mut(parameter_id) {
            state.touching = false;
            match state.mode {
                AutomationMode::Touch => {
                    state.manual_value = None;
                }
                AutomationMode::Latch => {
                    state.latched = true;
                    // manual_value remains set
                }
                AutomationMode::Read => {
                    state.manual_value = None;
                }
            }
        }
    }

    /// Reset latch for a parameter (resume reading automation curve).
    pub fn reset_latch(&mut self, parameter_id: &str) {
        if let Some(state) = self.states.get_mut(parameter_id) {
            state.latched = false;
            if !state.touching {
                state.manual_value = None;
            }
        }
    }

    // -- rendering ----------------------------------------------------------

    /// Render automation values for a processing block.
    ///
    /// `start_sample` is the absolute sample position of the first sample.
    /// `block_size` is the number of samples in the buffer.
    ///
    /// After calling this, retrieve per-sample values via [`Self::value_at`]
    /// or [`Self::rendered_block`].
    pub fn render_block(&mut self, start_sample: u64, block_size: usize) {
        if !self.enabled {
            self.rendered.clear();
            return;
        }

        let lane_ids: Vec<String> = self.lanes.keys().cloned().collect();

        for id in &lane_ids {
            let lane = match self.lanes.get(id) {
                Some(l) => l,
                None => continue,
            };
            let state = match self.states.get(id) {
                Some(s) => s,
                None => continue,
            };

            let mut values = Vec::with_capacity(block_size);

            for i in 0..block_size {
                let sample_pos = start_sample.saturating_add(i as u64);
                let v = self.resolve_value(lane, state, sample_pos);
                values.push(v);
            }

            self.rendered.insert(id.clone(), values);
        }
    }

    /// Resolve the effective value for a sample position, taking mode into
    /// account.
    fn resolve_value(
        &self,
        lane: &AutomationLaneData,
        state: &LaneState,
        sample_pos: u64,
    ) -> f32 {
        match state.mode {
            AutomationMode::Read => lane.evaluate(sample_pos),
            AutomationMode::Touch => {
                if state.touching {
                    state
                        .manual_value
                        .unwrap_or_else(|| lane.evaluate(sample_pos))
                        .clamp(lane.min_value, lane.max_value)
                } else {
                    lane.evaluate(sample_pos)
                }
            }
            AutomationMode::Latch => {
                if state.touching || state.latched {
                    state
                        .manual_value
                        .unwrap_or_else(|| lane.evaluate(sample_pos))
                        .clamp(lane.min_value, lane.max_value)
                } else {
                    lane.evaluate(sample_pos)
                }
            }
        }
    }

    /// Get the rendered value for a parameter at a sample offset within the
    /// most recently rendered block.
    ///
    /// Returns `None` if the parameter has no rendered data at that offset.
    #[must_use]
    pub fn value_at(&self, parameter_id: &str, sample_offset: usize) -> Option<f32> {
        self.rendered
            .get(parameter_id)
            .and_then(|v| v.get(sample_offset).copied())
    }

    /// Get the entire rendered block for a parameter.
    #[must_use]
    pub fn rendered_block(&self, parameter_id: &str) -> Option<&[f32]> {
        self.rendered.get(parameter_id).map(|v| v.as_slice())
    }

    /// Clear all lanes, state, and rendered data.
    pub fn clear(&mut self) {
        self.lanes.clear();
        self.states.clear();
        self.rendered.clear();
    }
}

impl Default for AutomationEngine {
    fn default() -> Self {
        Self::new(AutomationMode::Read)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_lane(id: &str, points: &[(u64, f32)]) -> AutomationLaneData {
        let mut lane = AutomationLaneData::new(id, 0.0, 1.0);
        for &(pos, val) in points {
            lane.add_breakpoint(AutomationBreakpoint::new(pos, val));
        }
        lane
    }

    // -- lane evaluation ----------------------------------------------------

    #[test]
    fn test_lane_linear_interpolation() {
        let lane = make_lane("gain", &[(0, 0.0), (100, 1.0)]);
        let mid = lane.evaluate(50);
        assert!(
            (mid - 0.5).abs() < 1e-4,
            "Linear midpoint should be 0.5, got {mid}"
        );
    }

    #[test]
    fn test_lane_before_first_returns_first_value() {
        let lane = make_lane("gain", &[(100, 0.7), (200, 1.0)]);
        let v = lane.evaluate(0);
        assert!(
            (v - 0.7).abs() < 1e-5,
            "Before first breakpoint should return first value, got {v}"
        );
    }

    #[test]
    fn test_lane_after_last_returns_last_value() {
        let lane = make_lane("gain", &[(0, 0.0), (100, 0.8)]);
        let v = lane.evaluate(500);
        assert!(
            (v - 0.8).abs() < 1e-5,
            "After last breakpoint should return last value, got {v}"
        );
    }

    #[test]
    fn test_lane_value_clamped() {
        let mut lane = AutomationLaneData::new("x", 0.0, 0.5);
        lane.add_breakpoint(AutomationBreakpoint::new(0, 1.0));
        let v = lane.evaluate(0);
        assert!(
            (v - 0.5).abs() < 1e-5,
            "Value should be clamped to max, got {v}"
        );
    }

    // -- engine basic -------------------------------------------------------

    #[test]
    fn test_engine_add_remove_lane() {
        let mut engine = AutomationEngine::default();
        engine.add_lane(make_lane("vol", &[(0, 0.5)]));
        assert_eq!(engine.lane_count(), 1);
        engine.remove_lane("vol");
        assert_eq!(engine.lane_count(), 0);
    }

    #[test]
    fn test_engine_read_mode_renders() {
        let mut engine = AutomationEngine::new(AutomationMode::Read);
        engine.add_lane(make_lane("gain", &[(0, 0.0), (100, 1.0)]));
        engine.render_block(0, 101);

        let v0 = engine.value_at("gain", 0);
        let v50 = engine.value_at("gain", 50);
        let v100 = engine.value_at("gain", 100);
        assert!(v0.is_some());
        assert!((v0.unwrap_or(-1.0) - 0.0).abs() < 1e-4);
        assert!((v50.unwrap_or(-1.0) - 0.5).abs() < 1e-3);
        assert!((v100.unwrap_or(-1.0) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_engine_disabled_no_render() {
        let mut engine = AutomationEngine::new(AutomationMode::Read);
        engine.set_enabled(false);
        engine.add_lane(make_lane("gain", &[(0, 0.5)]));
        engine.render_block(0, 64);
        assert!(engine.value_at("gain", 0).is_none());
    }

    // -- touch mode ---------------------------------------------------------

    #[test]
    fn test_touch_mode_override_and_release() {
        let mut engine = AutomationEngine::new(AutomationMode::Touch);
        engine.add_lane(make_lane("pan", &[(0, 0.0), (100, 1.0)]));

        // Touch the control with a manual value of 0.75.
        engine.touch_begin("pan", 0.75);
        engine.render_block(0, 10);
        let v = engine.value_at("pan", 5);
        assert!(
            (v.unwrap_or(-1.0) - 0.75).abs() < 1e-5,
            "Touch override should produce manual value"
        );

        // Release — should revert to recorded curve.
        engine.touch_end("pan");
        engine.render_block(50, 1);
        let v = engine.value_at("pan", 0);
        assert!(
            (v.unwrap_or(-1.0) - 0.5).abs() < 1e-3,
            "After touch release, should revert to curve at sample 50"
        );
    }

    // -- latch mode ---------------------------------------------------------

    #[test]
    fn test_latch_mode_holds_after_release() {
        let mut engine = AutomationEngine::new(AutomationMode::Latch);
        engine.add_lane(make_lane("vol", &[(0, 0.0), (100, 1.0)]));

        // Touch and set 0.3.
        engine.touch_begin("vol", 0.3);
        engine.touch_end("vol");

        // After release, value should be latched at 0.3.
        engine.render_block(80, 1);
        let v = engine.value_at("vol", 0);
        assert!(
            (v.unwrap_or(-1.0) - 0.3).abs() < 1e-5,
            "Latch should hold last manual value, got {:?}",
            v
        );

        // Reset latch — should go back to curve.
        engine.reset_latch("vol");
        engine.render_block(50, 1);
        let v = engine.value_at("vol", 0);
        assert!(
            (v.unwrap_or(-1.0) - 0.5).abs() < 1e-3,
            "After latch reset should revert to curve"
        );
    }

    // -- rendered_block -----------------------------------------------------

    #[test]
    fn test_rendered_block_returns_full_slice() {
        let mut engine = AutomationEngine::default();
        engine.add_lane(make_lane("x", &[(0, 0.0), (10, 1.0)]));
        engine.render_block(0, 11);
        let block = engine.rendered_block("x");
        assert!(block.is_some());
        assert_eq!(block.map(|b| b.len()), Some(11));
    }

    // -- touch_update -------------------------------------------------------

    #[test]
    fn test_touch_update_changes_value() {
        let mut engine = AutomationEngine::new(AutomationMode::Touch);
        engine.add_lane(make_lane("send", &[(0, 0.0)]));

        engine.touch_begin("send", 0.4);
        engine.touch_update("send", 0.9);
        engine.render_block(0, 1);
        let v = engine.value_at("send", 0);
        assert!(
            (v.unwrap_or(-1.0) - 0.9).abs() < 1e-5,
            "touch_update should change override value"
        );
    }

    // -- per-lane mode override ---------------------------------------------

    #[test]
    fn test_per_lane_mode_override() {
        let mut engine = AutomationEngine::new(AutomationMode::Read);
        engine.add_lane(make_lane("gain", &[(0, 0.0), (100, 1.0)]));
        engine.set_lane_mode("gain", AutomationMode::Touch);

        engine.touch_begin("gain", 0.42);
        engine.render_block(50, 1);
        let v = engine.value_at("gain", 0);
        assert!(
            (v.unwrap_or(-1.0) - 0.42).abs() < 1e-5,
            "Per-lane Touch override should apply"
        );
    }

    // -- empty lane default value -------------------------------------------

    #[test]
    fn test_empty_lane_returns_midpoint() {
        let lane = AutomationLaneData::new("x", 0.2, 0.8);
        let v = lane.evaluate(42);
        assert!(
            (v - 0.5).abs() < 1e-5,
            "Empty lane should return midpoint of range, got {v}"
        );
    }

    // -- clear engine -------------------------------------------------------

    #[test]
    fn test_clear_engine() {
        let mut engine = AutomationEngine::default();
        engine.add_lane(make_lane("a", &[(0, 0.5)]));
        engine.add_lane(make_lane("b", &[(0, 0.5)]));
        engine.clear();
        assert_eq!(engine.lane_count(), 0);
    }
}
